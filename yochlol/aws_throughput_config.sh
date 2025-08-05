# Test single-threaded
#time aws s3 cp s3://senseye-ptsd/public/ptsd_ios/0619ebeb-cb69-4ff1-8527-645bb95b7241_1749687553530/0619ebeb-cb69-4ff1-8527-645bb95b7241_1749689124025_face_pairs_task_10.mp4 .

# Test with different concurrency levles
#aws configure set default.s3.max_concurrent_requests 5
#time aws s3 sync s3://senseye-ptsd/public/ptsd_ios/0619ebeb-cb69-4ff1-8527-645bb95b7241_1749687553530/ ./downloads/

# aws configure set default.s3.max_concurrent_requests 10
# aws configure set default.s3.max_bandwidth 1GB/s

##################
# Conservative settings to start
aws configure set default.s3.max_concurrent_requests 10
aws configure set default.s3.max_bandwidth 75MB/s
aws configure set default.s3.multipart_threshold 64MB
aws configure set default.s3.multipart_chunksize 16MB
aws configure set default.s3.use_accelerate_endpoint false # unavailable for this bucket
