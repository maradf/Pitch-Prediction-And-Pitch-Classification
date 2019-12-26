# Written by Mara Fennema (with help from David Weenink)
#
# This praatscript takes all the wav-files from a folder, splits the sounds into
# separate sounds, with the silences in the original sound as the lines on where
# to split. After doing this, the pitch is extracted, turned into a pitchtier,
# which get's interpolated quadratically, in order to have consistent timesteps
# between each pitch-value. The eventual pitchtier is then saved as a so-called
# headerless spreadsheetfile in the folder cointaining this file.
# This headerless spreadsheetfile created by praat which still has a header on
# first line.
# For best results when dealing with a large amount of files at the same time,
# run this file directly from the command line using the command
# praat splitAudioAndGetInterpolatedPitch.praat
# and do not run within the user interface of Praat.

# Define the path of which directory contains all the audio files.
# For best results, use a folder that is a direct child of the folder that this
# file is saved in.
directory$ = "PATH"

# Collect all the files
list = Create Strings as file list: "fileList", directory$ + "*.wav"
numberOfFiles = Get number of strings
for ifile to numberOfFiles
	writeInfoLine: "Total number of files is: ", numberOfFiles
	appendInfoLine: "Current file is: ", ifile
    selectObject: list
    filename$ = Get string: ifile
    sound = Read from file: directory$ + filename$

	# Create TextGrids and split audio into separate Sounds
	textGrid = To TextGrid (silences): 100, 0.0, -25, 0.2, 0.2,
        ... "silent", "sounding"
	selectObject: sound, textGrid
        Extract intervals where: 1, "yes", "is equal to", "sounding"

	# Create collection of all Sounds
	n = numberOfSelected ("Sound")
	for i to n
		sound [i] = selected ("Sound", i)
	endfor

	for i to n
		# Get Pitch
		selectObject: sound [i]
		pitch = To Pitch: 0.0, 75.0, 600.0

		# Get PitchTier and remove Pitch objects from list
		selectObject: pitch
		pitchtier = Down to PitchTier
		selectObject: pitchtier
		removeObject: pitch

		# Interpolate PitchTiers Quadratically
		Interpolate quadratically: 4, "Semitones"

		# Set as TableOfReal and remove Pitchtier objects from list
		tableOfReal = nocheck Down to TableOfReal: "Hertz"
		if tableOfReal <> undefined
			writeInfoLine: "Total number of files is: ", numberOfFiles
			appendInfoLine: "Current file is: ", ifile
			appendInfoLine: "Saving sounding part ", i, " out of ", n
			selectObject: tableOfReal
			removeObject: pitchtier

			# Save TableOfReal as headerless spreadsheet file
			objectname$ = selected$ ("TableOfReal")
			Save as headerless spreadsheet file: objectname$
		endif
	endfor
	# Prevents reaching of maximum amount of objects in praat
	select all
	minusObject: list
	Remove
endfor

writeInfoLine: "Hooray! You're done! You've done ", numberOfFiles, " audiofiles! That's a lot!"
