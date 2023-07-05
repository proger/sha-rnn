echo "=== Acquiring datasets ==="
echo "---"

mkdir -p data
cd data

echo "- Downloading enwik8 (Character)"
mkdir -p enwik8
cd enwik8
wget --continue http://mattmahoney.net/dc/enwik8.zip
python prep_enwik8.py
cd ..

echo "---"
echo "Happy language modeling :)"
