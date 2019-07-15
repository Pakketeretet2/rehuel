CC = g++
FLAGS = -std=c++11 -pedantic -g -O3 -ffast-math -fopenmp -march=native \
        -Werror=return-type -Werror=uninitialized -Wall 


MATRIX_VECTOR_DIR = /home/stefan/projects/matrix_vector/
CATCH2_DIR="/home/stefan/projects/Catch/single_include/catch2"
REHUEL_DIR="../"
TIMER_DIR="../my_timer/"

LNK = -L./ -larmadillo -lrehuel
INC = -I./ -I$(TIMER_DIR) -I$(REHUEL_DIR) -I$(CATCH2_DIR) -I$(MATRIX_VECTOR_DIR)

COMP = $(CC) $(FLAGS) $(INC)
LINK = $(CC) $(FLAGS) $(INC) $(LNK)

EXE = rehuel_test
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
