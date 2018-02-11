.PHONY: installDependencies
installDependencies:
		sudo apt-get install ffmpeg
		sudo pip install pyyaml
		sudo pip install imutils
		sudo pip install scipy
		sudo pip install numpy
		sudo pip install nose

.SILENT:
