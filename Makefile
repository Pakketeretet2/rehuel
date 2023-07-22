# Default compiler and flags:
CC = clang++
FLAGS = -O2 -std=c++11 -pedantic -g -fPIC -shared -march=native -mtune=native \
        -Werror=return-type -Werror=uninitialized -Wall

ARMA_DIR =

# Linking and include directories:
LNK = -L./ -larmadillo -llapack -lblas
INC = -I./

# Short-hand for "compile" and "link":
COMP = $(CC) $(FLAGS) $(INC)
LINK = $(CC) $(FLAGS) $(INC) $(LNK)

# Output executable name
EXE = librehuel.so

# Automatically find all source files based on extension:
EXT = cpp
SRC = $(wildcard *.$(EXT))

# Uncomment for windows:
#MAKE_DIR = $(if exist $(1),,mkdir $(1))
#S=\\
# Uncomment for Linux and Unix-like:
MAKE_DIR = mkdir -p $(1)
S=/

# Define where the compiled objects go:
OBJ_DIR = obj
OBJ = $(SRC:%.$(EXT)=$(OBJ_DIR)$(S)%.o)
OBJ_DIRS = $(dir $(OBJ))

.PHONY: all help clean install

all : $(EXE)

install : $(EXE)
	cp $(EXE) /usr/local/lib

$(OBJ_DIR) :
	$(call $(MAKE_DIR),$@)

help :
	@echo "SRC is $(SRC)"
	@echo "OBJ is $(OBJ)"

$(EXE) : $(OBJ)
	$(LINK) $(OBJ) -o $@

# Make sure that any sub-paths exist, then compile
# the source to object file, and generate dependency file:
$(OBJ_DIR)$(S)%.o : %.$(EXT)
	$(call MAKE_DIR,$(dir $@))
	$(COMP) -c $< -o $@

clean:
	rm -r $(OBJ_DIR)
	rm -f $(EXE)
