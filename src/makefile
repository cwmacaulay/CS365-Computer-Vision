# OSX compiler
CC = clang++

# Dwarf compiler
# CC = g++

CXX = $(CC)

# OSX include paths
CFLAGS = -I/opt/local/include -I../include

# Dwarf include paths
# CFLAGS = -I../include # opencv includes are in /usr/include
# CXXFLAGS = $(CFLAGS)

# OSX Library paths
LDFLAGS = -L/opt/local/lib

# Dwarf Library paths
# LDFLAGS = -L/usr/lib64 # opencv libraries are here

# OSX libraries
LDLIBS = -lopencv_core -lopencv_highgui -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_ml

# Dwarf libraries
# LDLIBS = -lopencv_core -lopencv_highgui -lopencv_video -lopencv_imgproc -lopencv_objdetect

BINDIR = ../bin

vid: vidDisplay.o
	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)

imod: imgMod.o
	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)

CBIR: CBIR.o
	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)

OR: OR.o
	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)

clean:
	rm -f *.o *~ 
