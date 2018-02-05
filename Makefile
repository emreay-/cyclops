.PHONY: installDependencies
installDependencies:
		pip install pyyaml
		pip install imutils
		pip install scipy
		pip install numpy
		sudo apt-get install ffmpeg

.SILENT:
