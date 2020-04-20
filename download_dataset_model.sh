#!/bin/sh
set -eu

#--------------
# In-shop Retrieve Dataset
#--------------
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=19YIkcGkaDHcROUm8H8_12_3fAokwHbg3' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=19YIkcGkaDHcROUm8H8_12_3fAokwHbg3" -O In-shop.zip

unzip In-shop.zip -d data/
rm In-shop.zip

#--------------
# In-shop Retrieve Pretrained Model
#--------------
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1HZ13jijnjXxQ4nnsiss-UZ7bxHLN0kjw' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1HZ13jijnjXxQ4nnsiss-UZ7bxHLN0kjw" -O epoch_100.pth

mkdir -p checkpoint/Retrieve/resnet/
mv epoch_100.pth checkpoint/Retrieve/resnet/

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1J3FmP5iVE-arwQZKP2QVrOwDTtTvlzJZ' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1J3FmP5iVE-arwQZKP2QVrOwDTtTvlzJZ" -O epoch_100.pth

mkdir -p checkpoint/Retrieve/vgg/global
mv epoch_100.pth checkpoint/Retrieve/vgg/global/

#--------------
# Resnet50
#--------------
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1_nyFqCu0K7-l-Qeno4XEJ6aV4jl_Muck' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1_nyFqCu0K7-l-Qeno4XEJ6aV4jl_Muck" -O resnet50.pth

mv resnet50.pth checkpoint/

#--------------
# vgg16
#--------------
wget 'https://download.pytorch.org/models/vgg16-397923af.pth' -O vgg16.pth
mv vgg16.pth checkpoint/

