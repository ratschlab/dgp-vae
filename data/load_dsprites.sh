DATA_DIR="data/dsprites"

mkdir -p ${DATA_DIR}
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1RxWbKPNST6oWQ1LGPCr7aSpSQrtkKzZU' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1RxWbKPNST6oWQ1LGPCr7aSpSQrtkKzZU" -O ${DATA_DIR}/dsprites_5000.npz && rm -rf /tmp/cookies.txt

# WARNING LINK DOES NOT WORK YET
