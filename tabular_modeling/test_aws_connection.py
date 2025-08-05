#!/usr/bin/env python3
"""
Test AWS connectivity for the pipeline
"""

import boto3
import awswrangler as wr
import sys

def test_aws_credentials():
    """Test basic AWS credential access"""
    print("1. Testing AWS credentials...")
    try:
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        print(f"   ✓ AWS Account: {identity['Account']}")
        print(f"   ✓ AWS User/Role: {identity['Arn']}")
        return True
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

def test_s3_access():
    """Test S3 bucket access"""
    print("\n2. Testing S3 access...")
    try:
        s3 = boto3.client('s3')
        
        # Test bucket access
        s3.head_bucket(Bucket='senseye-data-quality')
        print("   ✓ Can access senseye-data-quality bucket")
        
        # List a few objects
        response = s3.list_objects_v2(
            Bucket='senseye-data-quality',
            Prefix='messy_prototyping_saturn_uploads/',
            MaxKeys=5
        )
        
        if 'Contents' in response:
            print(f"   ✓ Found {len(response['Contents'])} sample objects")
        else:
            print("   ! No objects found in the expected path")
            
        return True
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

def test_athena_access():
    """Test Athena and database access"""
    print("\n3. Testing Athena access...")
    try:
        # List databases
        databases = wr.catalog.databases()
        print(f"   ✓ Found {len(databases)} databases")
        
        if 'data_quality' in databases['Database'].values:
            print("   ✓ data_quality database exists")
            
            # Check if PCL table exists
            tables = wr.catalog.tables(database='data_quality')
            if 'mp_pcl_scores' in tables['Table'].values:
                print("   ✓ mp_pcl_scores table exists")
                
                # Try a simple query
                query = "SELECT COUNT(*) as count FROM data_quality.mp_pcl_scores LIMIT 1"
                df = wr.athena.read_sql_query(query, database='data_quality')
                count = df['count'].iloc[0]
                print(f"   ✓ PCL table has {count} records")
            else:
                print("   ✗ mp_pcl_scores table not found")
                print("     Available tables:", tables['Table'].tolist()[:5])
        else:
            print("   ✗ data_quality database not found")
            print("     Available databases:", databases['Database'].tolist()[:5])
            
        return True
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("AWS Connection Test for Tabular Modeling Pipeline")
    print("=" * 50)
    
    all_passed = True
    
    # Test credentials
    if not test_aws_credentials():
        all_passed = False
        print("\n⚠️  AWS credentials not configured!")
        print("Please run: aws configure")
        sys.exit(1)
    
    # Test S3
    if not test_s3_access():
        all_passed = False
        print("\n⚠️  S3 access failed!")
        print("Check that you have permissions for senseye-data-quality bucket")
    
    # Test Athena
    if not test_athena_access():
        all_passed = False
        print("\n⚠️  Athena access failed!")
        print("Check that you have permissions for Athena and Glue catalog")
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✅ All tests passed! Ready to run the pipeline.")
        print("\nRun: ./run_full_pipeline.sh")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main()