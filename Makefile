.PHONY: installDependencies
installDependencies:
		sudo pip3 install pyyaml
		sudo pip3 install imutils
		sudo pip3 install scipy
		sudo pip3 install numpy
		sudo pip3 install nose
		sudo apt-get install ffmpeg

.SILENT:
