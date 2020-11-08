from CreateMaskForImage import readImage, createInitialMask, erosionOfMask, findContours
from CreateMaskForImage import findIndividualGrains, drawBoundingRects
from RunInference import demo_results
import os


def starter(model):
    originalimage = readImage(f"temp/" + os.listdir("temp")[0])
    print(type(originalimage))
    mask = createInitialMask(originalimage)
    eroded_mask = erosionOfMask(mask)
    contours, areas = findContours(eroded_mask)
    response = findIndividualGrains(contours, areas, originalimage)
    print(response)
    # ['AVERAGE', 'BAD', 'EXCELLENT', 'GOOD', 'WORSE']}
    class_names = ['Average', 'Bad', 'Excellent', 'Good', 'Worse']
    listOfFlags = []
    for value, image in enumerate(os.listdir("test_images")):
        predicted_label = demo_results(
            model, f"test_images/{image}", class_names)
        listOfFlags.append(predicted_label)
    print(listOfFlags)
    Finalimage = drawBoundingRects(contours, listOfFlags, originalimage)
    return Finalimage
