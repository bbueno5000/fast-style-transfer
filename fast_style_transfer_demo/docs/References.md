# References

- A Neural Algorithm of Artistic Style.pdf
- Instance Normalization.pdf
- Perceptual Losses for Real-Time Style Transfer.pdf

## Instructions

1. create vm instance
2. sudo apt-get update
3. sudo apt-get upgrade
4. sudo apt-get install python-pip
5. sudo apt-get install git-all
6. sudo pip install virtualenv
7. virtualenv ~/floyd
8. source ~/floyd/bin/activate
9. floyd login -u bbueno5000
10. git clone https://github.com/floydhub/fast-style-transfer.git
11. floyd run --env tensorflow-0.12:py2 --data narenst/datasets/neural-style-transfer-pre-trained-models/1:models "python evaluate.py --allow-different-dimensions --checkpoint /models/rain_princess.ckpt --in-path ./images/ --out-path /output/"
12. download images
