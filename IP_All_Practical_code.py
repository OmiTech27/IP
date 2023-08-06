#Program to display image using read and write operation

import numpy as np 
import cv2 
print("This program read and write image") 
# Load an color imagein grayscale
img = cv2.imread('C:\\Users\\IM160\\Desktop\\Python Practicals\\Practical1\\bike.jpg',1)
#Display the image
cv2.imshow('image',img)
#key binding function
cv2.waitKey(0)
#Destroyed all window we created earlier. cv2.destroyAllWindows()
# Write the image on the same directory
cv2.imwrite("C:\\Users\\IM160\Desktop\\Python Practicals\\Practical1\\grayscale.jpg", img); 

#Program to enhance image arithmetic and logical operations.
" import cv2
print("Program to preform arithmatic operation on image")
img1=cv2.imread('C:\\Users\\91875\\OneDrive\\Desktop\\img1.jpeg',1)
img2=cv2.imread('C:\\Users\\91875\\OneDrive\\Desktop\\img2.jpeg',1)
#Displey an input images
cv2.imshow('Input image 1',img1)
cv2.imshow('Input image 2',img2)
#Addition of two images
image_add=cv2.add(img1,img2)
cv2.imshow('Addition',image_add)
#Substraction image_sub=cv2.subtract(img2,img1)
cv2.imshow('Substraction',image_sub) 
#Multiply
image_mult=cv2.multiply(img2,img1)
cv2.imshow('Multiply',image_sub)
cv2.waitKey(0) cv2.destroyAllWindows() 

#2.2 Program to perform logical operation on image. Source
import cv2
print("Program to perform arithmetic operations on image")
# reading an image img1=cv2.imread('C:\\Users\\91875\\OneDrive\\Desktop\\img1.jpeg',1)
img2=cv2.imread('C:\\Users\\91875\\OneDrive\\Desktop\\img2.jpeg',1)
#Displaying an image
cv2.imshow('image 1',img1)
cv2.imshow('image 2',img2) # logical
bitwise and operations
image_and=cv2.bitwise_and(img1,img2
) cv2.imshow('bitwise And',image_and)
# logical bitwise OR operation
image_or=cv2.bitwise_or(img1,img2)
cv2.imshow('bitwise Or',image_or)
# logical bitwise XOR operation
image_xor=cv2.bitwise_xor(img1,img2)
cv2.imshow('bitwise Xor',image_xor) #
logical bitwise NOT operation
image_not=cv2.bitwise_not(img1)
cv2.imshow('negation ',image_not)
#Key binding function cv2.waitKey(0)
#Destroy all previous window
cv2.destroyAllWindows() 

#: Program to Implement Image Negative
import cv2
print("Program to preform arithmatic operation on image")
img2=cv2.imread('C:\\Users\\91875\\OneDrive\\Desktop\\img2.jpeg',1
) cv2.imshow('Input Image',img2)
#Substract Input image from maximum intencity
img_negtive=255-img2 cv2.imshow('Negative
Image',img_negtive) cv2.waitKey(0)
cv2.destroyAllWindows()


#Program to implement Thesholding of an image
cv2
import numpy as np
print("Python program to illustrate simple thesholding type on image")
# path to input image is specified and image is loaded with imread command image1
= cv2.imread('C:\\Users\\91875\\OneDrive\\Desktop\\img1.jpeg',1)
# Displaying input image
cv2.imshow('Original image',image1)
# cv2.cvtColor is applied over the image input with applied parameters to convert the
image in grayscale img = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
# Displaying and writing Grayscale image cv2.imshow('Grayscale
image',img)
cv2.imwrite("C:\\Users\\91875\\OneDrive\\Desktop\\img1.jpeg", img)
# applying different thresholding techniques on the input image all pixels value above
120 will be set to 255 ret, thresh1 = cv2.threshold(img, 120, 255,
cv2.THRESH_BINARY) ret, thresh2 = cv2.threshold(img, 120, 255,
cv2.THRESH_BINARY_INV) ret, thresh3 = cv2.threshold(img, 120, 255,
cv2.THRESH_TRUNC) ret, thresh4 = cv2.threshold(img, 120, 255,
cv2.THRESH_TOZERO) ret, thresh5 = cv2.threshold(img, 120, 255,
cv2.THRESH_TOZERO_INV) # the window showing output images with
the corresponding thresholding techniques applied to the input images 
cv2.imshow('Binary Threshold', thresh1) cv2.imshow('Binary ThresholdInverted', thresh2) cv2.imshow('Truncated Threshold', thresh3)
cv2.imshow('Set to 0', thresh4) cv2.imshow('Set to 0 Inverted', thresh5)
# Writing ouput images
cv2.imwrite("C:\\Users\\91875\\OneDrive\\Desktop\\binary_threshold.jpg", thresh1)
cv2.imwrite("C:\\Users\\91875\\OneDrive\\Desktop\\binary_threshold_inverted.jpg
", thresh2)
cv2.imwrite("C:\\Users\\91875\\OneDrive\\Desktop\\truncated_threshold.jpg",
thresh3) cv2.imwrite("C:\\Users\\91875\\OneDrive\\Desktop\\Set_to_zero.jpg",
thresh4)
cv2.imwrite("C:\\Users\\91875\\OneDrive\\Desktop\\Set_to_zero_inverted.jpg",
thresh5)
#key binding function
cv2.waitKey(0)
#Destroyed all window we created earlier.
cv2.destroyAllWindows()


#Program to implement smoothing or averaging filter in spatial domain
import cv2
import numpy as np # read the image image =
cv2.imread('C:\\Users\\91875\\OneDrive\\Desktop\\img2.jpeg',1)
cv2.imshow('Original image',image) # apply the 3x3 mean filter on the
image kernel = np.ones((3,3),np.float32)/9 processed_image =cv2.filter2D(image,-1,kernel)
# display image cv2.imshow('Mean or Average Filter Processing',
processed_image)
# save image to disk
cv2.imwrite("C:\\Users\\91875\\OneDrive\\Desktop\\resulting_image.jpg", processed_image)
# pause the execution of the script until a key on the keyboard is pressed
cv2.waitKey(0)
#Destroyed all window we created earlier.
cv2.destroyAllWindows()

#Program to produce the Histogram, Equalized Histogram and Equalized image of an input image.

" import
cv2
from matplotlib import pyplot as plt
# reads an input image img =
cv2.imread('C:\\Users\\91875\\OneDrive\\Desktop\\img2.jpeg',1)
# find frequency of pixels in range 0-255 histr =
cv2.calcHist([img],[0],None,[256],[0,256])
# show the plotting graph of an image
plt.plot(histr) plt.show()
# alternative way to find histogram of an image
plt.hist(img.ravel(),256,[0,256]) plt.show()
# To produced Equalized image
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # It converts the image into grayscale
dst = cv2.equalizeHist(img) # Apply histogram equalization with the function
# Display both original & Equalized images
cv2.imshow('Source image', img)
cv2.imshow('Equalized Image', dst) #
Storing the equalized image
cv2.imwrite('C:\\Users\\91875\\OneDrive\\Desktop\\img2.jpeg',dst)
# pause the execution of the script until a key on the keyboard is pressed
cv2.waitKey(0)
#Destroyed all window we created earlier.
cv2.destroyAllWindows()

# Program for smooth an image using low pass filter in frequency domain.
import cv2 import numpy
as np
# Read image from system as grayscale
img = cv2.imread('C:\\Users\\91875\\OneDrive\\Desktop\\img2.jpeg', 0)
# Showing input image
cv2.imshow('Input image',img) #
Create a 5x5 Kernel kernel =
np.ones((5,5), np.float32)/25
# Apply convolution between image and 5x5 Kernel dst =
cv2.filter2D(img,-1, kernel) # Showing the resulting image
cv2.imshow('Image after LPF',dst) # Store LPF image as lpf.jpg
cv2.imwrite('C:\\Users\\91875\\OneDrive\\Desktop\\img2.jpeg', dst)
# pause the execution of the script until a key on the keyboard is pressed
cv2.waitKey(0)
#Destroyed all window we created earlier.
cv2.destroyAllWindows() 

#: Program for smooth an image using high pass filter in frequency domain.
import cv2 # Reading the
image
img = cv2.imread('C:\\Users\\91875\\OneDrive\\Desktop\\img1.jpeg', 1)
# subtract the original image with the blurred image #
after subtracting add 127 to the total result
hpf = img - cv2.GaussianBlur(img, (21, 21), 3)+127
# display both original image and filtered image cv2.imshow("Original",
img) cv2.imshow("High Passed Filter", hpf) # Storing the result of HPF
cv2.imwrite('C:\\Users\\91875\\OneDrive\\Desktop\\hpf.jpg', hpf) #
pause the execution of the script until a key on the keyboard is pressed
cv2.waitKey(0)
#Destroyed all window we created earlier.
cv2.destroyAllWindows()

#Program to find DFT/FFT Forward and Inverse Transform of Image
import cv2 as cv import numpy as
np from matplotlib import pyplot as
plt
# Create a picture
# src = np.ones((5, 5), dtype=np.uint8)*100
src = cv.imread('C:\\Users\\91875\\OneDrive\\Desktop\\img1.jpeg' ,0)
print("*"*100) print(src) print(src.shape) f = np.fft.fft2(src)
print("*"*100) print(f) fshift = np.fft.fftshift(f) print("*"*100)
print(fshift)
# Convert complex numbers into floating-point numbers for Fourier spectrum display fimg = np.log(np.abs(fshift))
print(fimg)
# Inverse Fourier transform ifshift= np.fft.ifftshift(fshift) 
# Convert complex numbers into floating-point numbers for Fourier spectrum display
if img = np.log(np.abs(ifshift)) if_img = np.fft.ifft2(ifshift) origin_img = np.abs(if_img)
# Image display plt.subplot(221), plt.imshow(src, "gray"),
plt.title('origin') plt.axis('off') plt.subplot(222), plt.imshow(fimg,
"gray"), plt.title('fourier_img') plt.axis('off') plt.subplot(223),
plt.imshow(origin_img, "gray"), plt.title('origin_img') plt.axis('off')
plt.subplot(224), plt.imshow(ifimg, "gray"), plt.title('ifimg')
plt.axis('off')
plt.show()


#Title: Program to find DCT forward and Inverse Transform of Image.
from scipy.fftpack import dct, idct
import cv2 # implement 2D DCT
def dct2(a):
return dct(dct(a.T, norm='ortho').T, norm='ortho')
# implement 2D IDCT
def idct2(a):
return idct(idct(a.T, norm='ortho').T, norm='ortho')
from skimage.io import imread from skimage.color
import rgb2gray import numpy as np import
matplotlib.pylab as plt
# read lena RGB image and convert to grayscale
im = rgb2gray(imread('C:\\Users\\91875\\OneDrive\\Desktop\\img2.jpeg' ,0))
imF = dct2(im) im1 = idct2(imF) 
# Showing all images
cv2.imshow('Original image',im)
cv2.imshow('Image afte DCT',imF)
cv2.imshow('Image after IDCT',im1)
# check if the reconstructed image is nearly equal to the original image np.allclose(im,
im1)
# True
# plot original and reconstructed images with matplotlib.pylab plt.gray()
plt.subplot(121), plt.imshow(im), plt.axis('on'), plt.title('original image', size=20)
plt.subplot(122), plt.imshow(im1), plt.axis('on'), plt.title('DCT & IDCT',
size=20) plt.show()
# pause the execution of the script until a key on the keyboard is pressed
cv2.waitKey(0)
#Destroyed all window we created earlier.
cv2.destroyAllWindows() 



#Title: Program to find Edges using Prewit/Sobel/Fri-chen/Robert operators
import cv2
import numpy as np
img = cv2.imread('C:\\Users\\91875\\OneDrive\\Desktop\\img3.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) img_gaussian
= cv2.GaussianBlur(gray,(3,3),0)
#sobel img_sobelx =
cv2.Sobel(img_gaussian,cv2.CV_8U,1,0,ksize=5) img_sobely
= cv2.Sobel(img_gaussian,cv2.CV_8U,0,1,ksize=5) img_sobel
= img_sobelx + img_sobely
#prewitt
kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]]) img_prewittx =
cv2.filter2D(img_gaussian, -1, kernelx) img_prewitty =
cv2.filter2D(img_gaussian, -1, kernely)
# Showing images
cv2.imshow("Original Image", img)
cv2.imshow("Sobel X", img_sobelx)
cv2.imshow("Sobel Y", img_sobely) 
cv2.imshow("Sobel", img_sobel)
cv2.imshow("Prewitt X",
img_prewittx) cv2.imshow("PrewittY", img_prewitty)
cv2.imshow("Prewitt",
img_prewittx + img_prewitty)
cv2.waitKey(0)
cv2.destroyAllWindows() 

#Title: Program to find edges using canny Edge Detection
import cv2
import numpy as np
img = cv2.imread('C:\\Users\\91875\\OneDrive\\Desktop\\img2.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) img_gaussian
= cv2.GaussianBlur(gray,(3,3),0)
#canny edge detection
img_canny = cv2.Canny(img,100,200)
cv2.imshow("Original Image", img) cv2.imshow("Canny",
img_canny)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Title: Program to implement Huffmann coding technique for image compression.
import re import
numpy as np from PIL
import Image import
cv2
print("Huffman Compression Program")
print("=======================================================
=
=========")
h = int(input("Enter 1 if you want to input an colour image file, 2 for default gray
scale case:")) if h == 1:
file = input("Enter the filename:")
my_string =
np.asarray(Image.open(file),np.uint8) shape =
my_string.shape a = my_string print ("Enetered
string is:",my_string) my_string =
str(my_string.tolist())
elif h == 2:
array = np.arange(0, 737280, 1, np.uint8)
my_string = np.reshape(array, (1024, 720))
print ("Enetered string is:",my_string)
a = my_string my_string =
str(my_string.tolist())
else:
print("You entered invalid input") # taking user input
letters = [] only_letters
= [] for letter in
my_string:
if letter not in letters:
frequency = my_string.count(letter) #frequency of each letter repetition
letters.append(frequency) letters.append(letter) only_letters.append(letter)
nodes = [] while len(letters)
> 0:
nodes.append(letters[0:2])
letters = letters[2:]
nodes.sort() huffman_tree
= []
# sorting according to frequency
huffman_tree.append(nodes) node
def combine_nodes(nodes):
pos = 0 newnode
= [] if len(nodes)
> 1: nodes.sort()
#Make each unique character as a leaf
nodes[pos].append("1") # assigning values 1 and 0
nodes[pos+1].append("0") combined_node1 =
(nodes[pos] [0] + nodes[pos+1] [0])
combined_node2 = (nodes[pos] [1] + nodes[pos+1] [1]) # combining the nodes to
generate pathways
newnode.append(combined_node1)
newnode.append(combined_node2) 
newnodes=[]
newnodes.append(newnode)
newnodes = newnodes + nodes[2:] nodes
= newnodes
huffman_tree.append(nodes)
combine_nodes(nodes)
return huffman_tree # huffman tree generation
newnodes = combine_nodes(nodes)
huffman_tree.sort(reverse = True) print("Huffman
tree with merged pathways:")
checklist = [] for level in
huffman_tree: for node in
level:
if node not in checklist:
checklist.append(node)
else:
level.remove(node)
count = 0 for level in
huffman_tree:
print("Level", count,":",level) #print huffman tree
count+=1 print()
letter_binary = [] if
len(only_letters) == 1:
lettercode = [only_letters[0], "0"] letter_binary.append(letter_code*len(my_string))
else:
for letter in
only_letters: code
="" for node in
checklist:
if len (node)>2 and letter in node[1]: #genrating binary code
code = code + node[2]
lettercode =[letter,code] letter_binary.append(lettercode)
print(letter_binary)
print("Binary code generated:")
for letter in letter_binary:
print(letter[0], letter[1])
bitstring ="" for character
in my_string:
for item in letter_binary:
if character in item:
bitstring = bitstring + item[1]
binary ="0b"+bitstring print("Your
message as binary is:")
# binary code generated
uncompressed_file_size = len(my_string)*7 compressed_file_size = len(binary)-2
print("Your original file size was", uncompressed_file_size,"bits. The compressed size
is:",compressed_file_size) print("This is a saving of ",uncompressed_file_sizecompressed_file_size,"bits") output = open("compressed.txt","w+")
print("Compressed file generated as compressed.txt") output =
open("compressed.txt","w+") print("Decoding. ..... ")
output.write(bitstring)
bitstring = str(binary[2:])
uncompressed_string =""
code ="" for digit in
bitstring: code =
code+digit
pos=0 #iterating and decoding for letter in letter_binary:
if code ==letter[1]:
uncompressed_string=uncompressed_string+letter_binary[pos] [0]
code="" pos+=1
print("Your UNCOMPRESSED data is:") if
h == 1:
temp = re.findall(r'\d+', uncompressed_string) res = list(map(int, temp)) res =
np.array(res) res = res.astype(np.uint8) res = np.reshape(res, shape) print(res)
print("Observe the shapes and input and output arrays are matching or not")
print("Input image dimensions:",shape) print("Output image
dimensions:",res.shape) data = Image.fromarray(res)
data.save('C:\\Users\\IM160\\Desktop\\Python Practicals\\Practical
13\huffman.jpg') if a.all() == res.all():
print("Success")
if h == 2:
temp = re.findall(r'\d+',
uncompressed_string) res = list(map(int,
temp)) print(res) res = np.array(res) res =
res.astype(np.uint8) res = np.reshape(res,
(1024, 720)) print(res)
data = Image.fromarray(res)
data.save('C:\\Users\\IM160\\Desktop\\Python Practicals\\Practical
13\huffman.jpg') print("Success") 


       

                                    
