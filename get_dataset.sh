# !!NOTE!! Make sure you have git-lfs installed on your system beforehand
# Clean up before installing
echo "Cleaning up c4 repository..."
rm -rf c4

# Clone dataset repository
echo "Cloning c4 repository..."
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/allenai/c4

# Download the TREC Health misinformation 2021 topics and qrels
echo "Cleaning up topics and qrels..."
rm -rf eval
mkdir eval

echo "Downloading topics and qrels..."
curl --progress-bar -L https://trec.nist.gov/data/misinfo/misinfo-resources-2021.tar.gz -o eval/misinfo-resources-2021.tar.gz

echo "Extracting topics and qrels..."
tar -xzf eval/misinfo-resources-2021.tar.gz -C eval

echo "Cleaning up, removing archive..."
rm eval/misinfo-resources-2021.tar.gz

# Run script for downloading relevant documents
echo "Downloading relevant documents..."
python ./docnos.py --c4-dir c4 --topics-dir eval/misinfo-resources-2021/topics/misinfo-2021-topics.xml --qrels-dir eval/misinfo-resources-2021/qrels/qrels-35topics.txt --n 5

# Run script for indexing the downloaded documents
echo "Cleaning up existing indexes..."
rm -rf index
echo "Indexing downloaded documents"
python ./index.py --data-dir ./data