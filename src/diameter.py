# if we have no index to compute the size of the mole on the picture as a scale we need the user to input it manually

def diameter(size): 
    if size > 6: # the diameter of a good mole should be less than 6 mm after that it is a melanoma
        print("The mole is large and can be source of cancer.")
    else:
        print("The mole is small and probably not a source of cancer.")

def main():
    size = 7
    diameter(size)

if __name__ == "__main__":
    main()