# Nom du programme
PROG := deformation

SRC_DIR := src
INC_DIR := include
MODEL_DIR := models
BUILD_DIR := build

# Localisation de gmsh (sdk, source ou autre)
GMSH_LIB_DIR := ./gmsh-sdk/lib
GMSH_INC_DIR := ./gmsh-sdk/include

# Librairies à linker
LDLIBS := -lgmsh -lm #-lopenblas


# Detect Windows MinGW (including TDM-GCC)
ifneq (,$(findstring MINGW,$(OS)))
    OS := Windows_NT
endif

ifeq ($(OS), Darwin)
    CC ?= clang
    LDFLAGS := -Wl,-rpath,$(realpath $(GMSH_LIB_DIR))
else ifeq ($(OS), Windows_NT)
    CC ?= gcc
    ifeq ($(shell $(CC) --version | grep -c "tdm"), 1)
        TDM_GCC := 1
    endif
    ifndef TDM_GCC
        LDFLAGS := -Wl,--enable-auto-import
    endif
else
    CC ?= gcc
    LDFLAGS := -Wl,-rpath,$(GMSH_LIB_DIR) #-fsanitize=address
endif

# Flags de compilation
CFLAGS := -O3 # -fsanitize=address
INC_DIRS := -I $(INC_DIR) -I $(GMSH_INC_DIR) # Chemins vers les dir `include`
LIB_DIR := -L $(GMSH_LIB_DIR) # Chemins vers les dir `lib`

SRC_FILES := $(wildcard $(SRC_DIR)/*.c)
MODEL_FILES := $(wildcard $(MODEL_DIR)/*.c)
OBJS = $(patsubst $(SRC_DIR)/%.c, $(BUILD_DIR)/%.o, $(SRC_FILES)) \
       $(patsubst $(MODEL_DIR)/%.c, $(BUILD_DIR)/%.o, $(MODEL_FILES))

# Compilation principale
all: | $(BUILD_DIR) $(PROG)

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR) || true

$(BUILD_DIR)/%.o: $(MODEL_DIR)/%.c | $(BUILD_DIR)
	@echo "Compiling model $<..."
	@$(CC) -g -c $(CFLAGS) $(INC_DIRS) $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	@echo "Compiling source $<..."
	@$(CC) -g -c $(CFLAGS) $(INC_DIRS) $< -o $@

# Link final
$(PROG): $(OBJS)
	@echo "Linking $(PROG)..."
	@$(CC) -g -o $@ $(OBJS) $(LIB_DIR) $(LDLIBS) $(LDFLAGS)

# Nettoyage
clean:
	rm -f $(PROG) $(OBJS)
