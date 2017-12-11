import matplotlib.pyplot as plt
import parse_openpose_json
import prepocessing
import procrustes3


def superimpose(input, model, input_image, model_image):

    input = prepocessing.unpad(input)
    model = prepocessing.unpad(model)

    (d, Z, m) = procrustes3.procrustes(input, model, False)


    # Zoeken naar laagste punt van lichaam (linker of rechter voet)
    # =>   Max van linker en recht voet (y-coordinaat)
    # => pose wordt aligned met laagste punt (puur translatie)
    voet_index = None

    if input[10][1] >= input[13][1]:
        voet_index = 10
    else:
        voet_index = 13

    print("laagste punt: " , input[voet_index][1])

    translatie_factor = input[voet_index][1] - Z[voet_index][1]
    print("trans: ", translatie_factor)

    Z[:,1] = Z[:,1] + translatie_factor

    markersize = 3
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(14, 6))
    ax1.imshow(plt.imread(input_image))
    ax1.set_title(input_image + '(input)')
    ax1.plot(*zip(*input), marker='o', color='r', ls='', label='model', ms=markersize)  # ms = markersize

    ax2.set_title(model_image + '(model)')
    ax2.imshow(plt.imread(model_image))
    ax2.plot(*zip(*model), marker='o', color='r', ls='', label='model', ms=markersize)  # ms = markersize

    ax3.set_title('(model sumperimposed on input)')
    ax3.imshow(plt.imread(input_image))
    ax3.plot(*zip(*Z), marker='o', color='r', ls='', label='model', ms=markersize)  # ms = markersize
    plt.show()

    return Z
