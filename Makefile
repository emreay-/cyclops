.PHONY: installDependencies
installDependencies:
		pip install pyyaml
		pip install imutils
		pip install scipy
		pip install numpy
		pip install nose
		sudo apt-get install ffmpeg

.SILENT:
