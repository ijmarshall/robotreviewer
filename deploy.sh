#!/bin/bash

echo "copying to temporary directory"
TMP=`mktemp -d`
cp -a . $TMP

echo "adding to tar file"
CURR=`pwd`
echo "removing previous build"
rm deploy.tar.gz
cd $TMP
tar --exclude "*.log" --exclude ".git/*" --exclude ".git" --exclude "target" --exclude "node_modules" -z -c -v -f $CURR/deploy.tar.gz .

echo "removing temporary directory"
rm -rf $TMP

cd $CURR

echo "building image"
docker build .
