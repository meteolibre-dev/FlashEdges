"""
GCP storage client for the FlashEdges inference pipeline.

Adapted from flashnet/backend/gcp_client.py. Handles:

  - listing files under the input prefix ``inference_h5_global`` of
    ``gs://eumetsat_mtg_preprocess`` (H5 inputs named
    ``global_live_YYYYMMDD_HHMM.h5``),
  - selecting the file with the **latest date** encoded in its filename
    (not the latest GCS object ``updated`` time, which can be skewed by
    copies/uploads),
  - downloading the input H5 to the local scratch dir,
  - downloading the model weights from GCS,
  - uploading the resulting GeoTIFF forecasts to the output bucket.
"""

import os
import re
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List
from dataclasses import dataclass

from google.cloud import storage
from google.oauth2 import service_account
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .config import get_config, GCPConfig

logger = logging.getLogger(__name__)

# global_live_20260817_1500.h5  ->  date token "20260817_1500"
_H5_DATE_RE = re.compile(r"global_live_(\d{8})_(\d{4})\.h5$")


def _h5_date_token(name: str) -> Optional[str]:
    """Return the YYYYMMDD_HHMM token encoded in an H5 filename, or None."""
    m = _H5_DATE_RE.search(name)
    if not m:
        return None
    return f"{m.group(1)}_{m.group(2)}"


@dataclass
class GCSFileInfo:
    """Information about a file in GCS."""
    name: str
    bucket: str
    size: int
    updated: datetime
    gcs_path: str


class GCPStorageClient:
    """Client for interacting with Google Cloud Storage."""

    def __init__(self, config: Optional[GCPConfig] = None):
        self.config = config or get_config().gcp
        self._client: Optional[storage.Client] = None

    def _get_client(self) -> storage.Client:
        if self._client is None:
            credentials = None
            if self.config.credentials_path and os.path.exists(self.config.credentials_path):
                credentials = service_account.Credentials.from_service_account_file(
                    self.config.credentials_path
                )
            elif os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
                credentials = service_account.Credentials.from_service_account_file(
                    os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
                )
            self._client = storage.Client(
                project=self.config.project_id,
                credentials=credentials,
            )
        return self._client

    def _get_bucket(self, bucket_name: str) -> storage.Bucket:
        return self._get_client().bucket(bucket_name)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
    )
    def download_file(
        self,
        source_gcs_path: str,
        dest_path: str,
        bucket_name: Optional[str] = None,
    ) -> str:
        """Download a file from GCS to a local path."""
        bucket_name = bucket_name or self.config.source_bucket
        bucket = self._get_bucket(bucket_name)

        path_parts = source_gcs_path.split("/")
        if path_parts[0] == bucket_name:
            blob_name = "/".join(path_parts[1:])
        else:
            blob_name = source_gcs_path

        Path(dest_path).parent.mkdir(parents=True, exist_ok=True)

        blob = bucket.blob(blob_name)
        logger.info(f"Downloading gs://{bucket_name}/{blob_name} to {dest_path}")
        blob.download_to_filename(dest_path)
        logger.info(f"Successfully downloaded {source_gcs_path}")
        return dest_path

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
    )
    def upload_file(
        self,
        source_path: str,
        dest_gcs_path: str,
        bucket_name: Optional[str] = None,
        content_type: Optional[str] = None,
    ) -> str:
        """Upload a local file to GCS."""
        bucket_name = bucket_name or self.config.dest_bucket
        bucket = self._get_bucket(bucket_name)

        path_parts = dest_gcs_path.split("/")
        if path_parts[0] == bucket_name:
            blob_name = "/".join(path_parts[1:])
        else:
            blob_name = dest_gcs_path

        blob = bucket.blob(blob_name)
        logger.info(f"Uploading {source_path} to gs://{bucket_name}/{blob_name}")
        if content_type:
            blob.upload_from_filename(source_path, content_type=content_type)
        else:
            blob.upload_from_filename(source_path)
        logger.info(f"Successfully uploaded to gs://{bucket_name}/{blob_name}")
        return dest_gcs_path

    def list_files(
        self,
        prefix: str = "",
        bucket_name: Optional[str] = None,
        extension: Optional[str] = None,
        max_results: int = 1000,
    ) -> List[GCSFileInfo]:
        """List files under a prefix.

        The configured ``source_prefix`` (``inference_h5_global``) is always
        prepended to the requested prefix.
        """
        bucket_name = bucket_name or self.config.source_bucket
        bucket = self._get_bucket(bucket_name)

        full_prefix = f"{self.config.source_prefix}/{prefix}".lstrip("/")
        logger.info(f"Listing files in gs://{bucket_name}/{full_prefix}")

        blobs = bucket.list_blobs(prefix=full_prefix, max_results=max_results)

        files = []
        for blob in blobs:
            if extension and not blob.name.endswith(extension):
                continue
            files.append(GCSFileInfo(
                name=os.path.basename(blob.name),
                bucket=bucket_name,
                size=blob.size,
                updated=blob.updated,
                gcs_path=f"{bucket_name}/{blob.name}",
            ))

        logger.info(f"Found {len(files)} files")
        return files

    def get_latest_file(self, bucket_name: Optional[str] = None) -> Optional[GCSFileInfo]:
        """Return the H5 input with the latest date encoded in its filename.

        Selection is by the ``global_live_YYYYMMDD_HHMM.h5`` date token
        (string sort == chronological sort because the token is zero-padded),
        falling back to the GCS ``updated`` time for files that don't match
        the naming convention.
        """
        files = self.list_files(bucket_name=bucket_name, extension=".h5")
        if not files:
            return None

        def sort_key(f: GCSFileInfo):
            token = _h5_date_token(f.name)
            # (has_token, token_or_empty, updated_iso) — tokenized files sort
            # after non-tokenized ones by their date token.
            if token is not None:
                return (1, token, "")
            return (0, "", f.updated.isoformat())

        files.sort(key=sort_key, reverse=True)
        return files[0]

    def download_model(self, gcs_path: str, local_path: str) -> str:
        """Download model weights from a gs:// path to a local path."""
        if not gcs_path.startswith("gs://"):
            raise ValueError(f"Invalid GCS path: {gcs_path}")
        path_parts = gcs_path[5:].split("/")
        bucket_name = path_parts[0]
        gcs_path_rel = "/".join(path_parts[1:])
        return self.download_file(gcs_path_rel, local_path, bucket_name=bucket_name)

    def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None


def get_gcs_client() -> GCPStorageClient:
    """Get a GCPStorageClient configured from the environment."""
    return GCPStorageClient()


def parse_h5_date(filename: str) -> str:
    """Extract the ``YYYYMMDD`` date folder token from an H5 filename.

    ``global_live_20260817_1500.h5`` -> ``20260817``.
    """
    token = _h5_date_token(filename)
    if token is None:
        raise ValueError(f"Cannot parse date from H5 filename: {filename}")
    return token.split("_")[0]
