# AWS Setup for PCL Score Fetching

## Prerequisites

1. **AWS CLI Installation**
   ```bash
   # Install AWS CLI (if not already installed)
   pip install awscli
   ```

2. **Configure AWS Credentials**
   ```bash
   aws configure
   ```
   You'll need to provide:
   - AWS Access Key ID
   - AWS Secret Access Key
   - Default region name (e.g., us-east-1)
   - Default output format (json)

3. **Install AWS Data Wrangler**
   ```bash
   pip install awswrangler
   ```

## Verify AWS Access

Before running the pipeline, verify your AWS access:

```bash
# Test S3 access
aws s3 ls s3://senseye-data-quality/ --max-items 5

# Test Athena access (list databases)
aws athena list-data-catalogs
```

## Required AWS Permissions

Your AWS IAM user/role needs the following permissions:

### For S3 Access:
- `s3:ListBucket` on `senseye-data-quality` bucket
- `s3:GetObject` on `senseye-data-quality/*`

### For Athena Access:
- `athena:StartQueryExecution`
- `athena:GetQueryExecution`
- `athena:GetQueryResults`
- `athena:GetDataCatalog`
- `athena:GetDatabase`
- `athena:GetTableMetadata`
- `athena:ListDataCatalogs`
- `athena:ListDatabases`
- `athena:ListTableMetadata`

### For Glue Catalog Access:
- `glue:GetDatabase`
- `glue:GetTable`
- `glue:GetPartitions`

### For S3 Query Results:
- `s3:PutObject` on Athena results bucket
- `s3:GetObject` on Athena results bucket
- `s3:ListBucket` on Athena results bucket

## Troubleshooting

1. **"Unable to locate credentials" error**
   - Run `aws configure` and enter your credentials
   - Or set environment variables:
     ```bash
     export AWS_ACCESS_KEY_ID=your_access_key
     export AWS_SECRET_ACCESS_KEY=your_secret_key
     export AWS_DEFAULT_REGION=us-east-1
     ```

2. **"Access Denied" errors**
   - Check that your IAM user has the required permissions listed above
   - Verify you're in the correct AWS account

3. **Athena query errors**
   - Ensure the `data_quality.mp_pcl_scores` table exists
   - Check that you have access to the `data_quality` database

## Running the Pipeline

Once AWS is configured:

```bash
# Install all dependencies
pip install -r requirements.txt

# Run the full pipeline
./run_full_pipeline.sh
```

The pipeline will:
1. Download session files from S3
2. Process them into a single dataframe
3. Fetch PCL scores from Athena
4. Merge PCL scores with session data
5. Run ML models for both PTSD prediction and PCL score regression