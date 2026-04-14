# ML Diagnostics and Profiling

MaxDiffusion supports automated profiling and performance tracking via [Google Cloud ML Diagnostics](https://docs.cloud.google.com/tpu/docs/ml-diagnostics/sdk).

## 1. Manual Installation
To keep the core MaxDiffusion repository lightweight and ensure it runs without dependencies for users who don't need profiling, the ML Diagnostics packages are **not** installed by default.

To use this feature, you must manually install the required package in your environment:
```bash
pip install google-cloud-mldiagnostics
```

## 2. Configuration Settings
To enable ML Diagnostics for your training or generation jobs, you need to update your configuration. You can either add these directly to your .yml config file or pass them as command-line arguments:

```yaml
# ML Diagnostics settings
enable_ml_diagnostics: True
profiler_gcs_path: "gs://<your-bucket-name>/profiler/ml_diagnostics"
enable_ondemand_xprof: True
```

## 3. GCS Bucket Permissions (Troubleshooting)
The GCS bucket you provide in `profiler_gcs_path` **must** have the correct IAM permissions to allow the Hypercompute Cluster service account to write data.

If permissions are not configured correctly, your job will fail with an error similar to this:
> `message: 'service-32478767326@gcp-sa-hypercomputecluster.iam.gserviceaccount.com does not have storage.buckets.get access to the GCS bucket <your-bucket>: permission denied'`

**Fix:** Ensure you grant the required Storage roles (e.g., `Storage Object Admin`) to the service account mentioned in your error message for your specific GCS bucket.

## 4. Viewing Your Runs
Once your job is running with diagnostics enabled, you can monitor the profiles, execution times, and metrics in the Cluster Director console here:

🔗 **https://pantheon.corp.google.com/cluster-director/diagnostics**