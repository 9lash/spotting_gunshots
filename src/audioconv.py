# Audio format conversion 

from os import path
from pydub import AudioSegment

# files                                                                         
src = "../pistol_shot.m4a"
dst = "../pistol_shot.wav"


def main():

	# convert mp3 to wav                                                            
	sound = AudioSegment.from_mp3(src)
	sound.export(dst, format="wav")

if "__name__" == "__main__":
	main()
	