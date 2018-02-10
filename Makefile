.PHONY: installDependencies
installDependencies:
		sudo apt-get install python3-pip
		sudo apt-get install ffmpeg
		sudo pip3 install pyyaml
		sudo pip3 install imutils
		sudo pip3 install scipy
		sudo pip3 install numpy
		sudo pip3 install nose

.SILENT:
