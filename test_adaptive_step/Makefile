CC = g++
FLAGS = -std=c++11 -pedantic -g -O3 -march=native -ffinite-math-only \
        -Werror=return-type -Werror=uninitialized -Wall

LNK = -L./ -L../ -larmadillo -lrehuel
INC = -I./ -I../ -I../my_timer/

COMP = $(CC) $(FLAGS) $(INC)
LINK = $(CC) $(FLAGS) $(INC) $(LNK)

EXE = test_adaptive_step
EXT = cpp
SRC = $(wildcard *.$(EXT))

# For windows:
#MAKE_DIR = $(if exist $(1),,mkdir $(1))
#S=\\
# Linux and Unix-like:
MAKE_DIR = mkdir -p $(1)
S=/



OBJ_DIR = obj
OBJ = $(SRC:%.$(EXT)=$(OBJ_DIR)$(S)%.o)
OBJ_DIRS = $(dir $(OBJ))
DEPS = $(OBJ:%.o=%.d)

.PHONY: dirs all help clean install

all : dirs $(EXE)

install : $(EXE)
	cp $(EXE) /usr/local/bin

dirs : $(OBJ_DIR)

$(OBJ_DIR) :
	$(call $(MAKE_DIR),$@)

help :
	@echo "SRC is $(SRC)"
	@echo "OBJ is $(OBJ)"
	@echo "DEPS is $(DEPS)"

$(EXE) : $(OBJ)
	$(LINK) $(OBJ) -o $@

$(OBJ_DIR)$(S)%.o : %.$(EXT)
	$(call MAKE_DIR,$(dir $@))
	$(COMP) -c $< -o $@
	$(COMP) -M -MT '$@' $< -MF $(@:%.o=%.d)

clean:
	rm -r $(OBJ_DIR)
	rm -f $(EXE)

-include $(DEPS)
