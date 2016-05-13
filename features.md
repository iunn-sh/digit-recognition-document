# Features
(Including the description and generating method of each feature)

![train/00601459](./gitbook/images/00601459.png)
*Preview of features (best training result = deskew)*

## Raw Data
Read data byte-by-byte then save as csv file

```python
def get_raw(fn):
    with open(fn, 'rb') as f:
        px = f.read(PIXEL_COUNT)
        line = []
        for i in range(PIXEL_COUNT):
            line.append(ord(px[i]))

    return line
```

## Deskew (used in final training)
Remove skew from raw data to make pixels left-right-balance [using its second order moments](http://docs.opencv.org/3.1.0/dd/d3b/tutorial_py_svm_opencv.html#gsc.tab=0)

```python
def gen_deskew(fn, output_fn):
    pixel = random_forest.read_csv_to_list(fn)
    deskew = []
    affine_flags = cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR

    for row in pixel:
        name = row[0]
        img = np.array(row[1:], dtype=np.float)
        img.shape = (LENGTH, LENGTH)
        img_deskew = []

        m = cv2.moments(img)
        if abs(m['mu02']) < 1e-2:
            img_deskew = img.copy()
        skew = m['mu11'] / m['mu02']
        M = np.float32([[1, skew, -0.5 * LENGTH * skew], [0, 1, 0]])
        img_deskew = cv2.warpAffine(
            img, M, (LENGTH, LENGTH), flags=affine_flags)

        linked = [name] + ['{}'.format(x) for x in img_deskew.flatten()]
        deskew.append(linked)

    with open(output_fn, 'wb') as f:
        for l in deskew:
            f.write(','.join(l) + '\n')
```

## Binary
Make deskew data binary to 0 or 255
```python
def gen_binary(fn, output_fn):
    pixel = random_forest.read_csv_to_list(fn)
    binary = []

    for row in pixel:
        name = row[0]
        img = np.array(row[1:], dtype=np.float)
        img.shape = (LENGTH, LENGTH)
        img = img.astype(np.uint8)

        img_binary = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 0)

        linked = [name] + ['{}'.format(x) for x in img_binary.flatten()]
        binary.append(linked)

    with open(output_fn, 'wb') as f:
        for l in binary:
            f.write(','.join(l) + '\n')
```

## Erosion
Erode deskew data, keep white pixel only when surrounding ones are also white
```python
def gen_erosion(fn, output_fn):
    pixel = random_forest.read_csv_to_list(fn)
    erosion = []

    for row in pixel:
        name = row[0]
        img = np.array(row[1:], dtype=np.float)
        img.shape = (LENGTH, LENGTH)
        img = img.astype(np.uint8)

        kernel = np.ones((3, 3), np.uint8)
        img_erosion = cv2.erode(img, kernel, iterations=1)

        linked = [name] + ['{}'.format(x) for x in img_erosion.flatten()]
        erosion.append(linked)

    with open(output_fn, 'wb') as f:
        for l in erosion:
            f.write(','.join(l) + '\n')
```

## Skeleton
Erode deskew data intil it is 1-pixel thick
```python
def gen_skeleton(fn, output_fn):
    pixel = random_forest.read_csv_to_list(fn)
    skeleton = []

    for row in pixel:
        name = row[0]
        img = np.array(row[1:], dtype=np.float)
        img.shape = (LENGTH, LENGTH)
        img = img.astype(np.uint8)

        size = np.size(img)
        img_skeleton = np.zeros(img.shape,np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
        done = False

        while(not done):
            eroded = cv2.erode(img,element)
            temp = cv2.dilate(eroded,element)
            temp = cv2.subtract(img,temp)
            img_skeleton = cv2.bitwise_or(img_skeleton,temp)
            img = eroded.copy()

            zeros = size - cv2.countNonZero(img)
            if zeros==size:
                done = True

        linked = [name] + ['{}'.format(x) for x in img_skeleton.flatten()]
        skeleton.append(linked)

    with open(output_fn, 'wb') as f:
        for l in skeleton:
            f.write(','.join(l) + '\n')
```

## Contour
Calculate contour points around boundary
```python
def gen_contour(fn, output_fn):
    pixel = random_forest.read_csv_to_list(fn)
    contour = []

    for row in pixel:
        name = row[0]
        img = np.array(row[1:], dtype=np.float)
        img.shape = (LENGTH, LENGTH)
        img = img.astype(np.uint8)

        ret, thresh = cv2.threshold(img, 127, 255, 0)
        cnts, hierarchy = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)
        for c in cnts:
            cv2.drawContours(img, c, -1, (0, 255, 0), 3)

        flatten = [
            val for lists in cnts for sublists in lists for sublist in sublists for val in sublist]
        linked = [name] + ['{}'.format(x) for x in flatten]
        contour.append(linked)

    with open(output_fn, 'wb') as f:
        for l in contour:
            f.write(','.join(l) + '\n')
```

## Bounding Box
Calculate bounding box with
* box center point
* box width
* box height
* box width/height ratio

```python
def gen_bounding_box(fn, output_fn):
    pixel = random_forest.read_csv_to_list(fn)
    bounding_box = []

    for row in pixel:
        name = row[0]
        img = np.array(row[1:], dtype=np.float)
        img.shape = (LENGTH, LENGTH)
        img = img.astype(np.uint8)

        B = np.argwhere(img)
        (ystart, xstart), (ystop, xstop) = B.min(0), B.max(0) + 1
        width = xstop - xstart
        height = ystop - ystart
        center_x = (xstop + xstart) / 2
        center_y = (ystop + ystart) / 2
        ratio = float(width) / float(height)

        flatten = [center_x, center_y, width, height, ratio]
        linked = [name] + ['{}'.format(x) for x in flatten]
        bounding_box.append(linked)

    with open(output_fn, 'wb') as f:
        for l in bounding_box:
            f.write(','.join(l) + '\n')
```
