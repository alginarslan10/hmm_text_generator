#!/bin/bash

#Check if docker repo exists and if it is installed
PKG_INSTALLED =$(dpkg-query -W -f='${Status}' docker 2>/dev/null| grep -c "ok installed")

if ! grep -q "$DOCKER_REPO" /etc/apt/sources.list && ! $PKG_INSTALLED; then
	echo "$DOCKER_REPO" >> /etc/apt/sources.list

	if !$PKG_INSTALLED; then
		apt remove docker docker-engine docker.io containerd runc
		apt update
		apt install ca-certificates curl gnupg lsb-release
		curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
		echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
		apt update
		apt install docker-ce docker-ce-cli containerd.io
		apt install --only-upgrade *docker*
	fi
fi

#Add user to docker
usermod -aG docker $USER
systemctl restart docker

#Install Zemberek-Server
docker pull cbilgili/zemberek-nlp-server:latest
docker run -p 4567:4567 cbilgili/zemebrek-nlp-server:latest

#Install python requirements
pip3 install -r requirements.txt
