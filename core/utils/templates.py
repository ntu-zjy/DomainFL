
cars_template = [
    lambda c: f'a photo of a {c}.',
    lambda c: f'a photo of the {c}.',
    lambda c: f'a photo of my {c}.',
    lambda c: f'i love my {c}!',
    lambda c: f'a photo of my dirty {c}.',
    lambda c: f'a photo of my clean {c}.',
    lambda c: f'a photo of my new {c}.',
    lambda c: f'a photo of my old {c}.',
]

cifar10_template = [
    lambda c: f'a photo of a {c}.',
    lambda c: f'a blurry photo of a {c}.',
    lambda c: f'a black and white photo of a {c}.',
    lambda c: f'a low contrast photo of a {c}.',
    lambda c: f'a high contrast photo of a {c}.',
    lambda c: f'a bad photo of a {c}.',
    lambda c: f'a good photo of a {c}.',
    lambda c: f'a photo of a small {c}.',
    lambda c: f'a photo of a big {c}.',
    lambda c: f'a photo of the {c}.',
    lambda c: f'a blurry photo of the {c}.',
    lambda c: f'a black and white photo of the {c}.',
    lambda c: f'a low contrast photo of the {c}.',
    lambda c: f'a high contrast photo of the {c}.',
    lambda c: f'a bad photo of the {c}.',
    lambda c: f'a good photo of the {c}.',
    lambda c: f'a photo of the small {c}.',
    lambda c: f'a photo of the big {c}.',
]

cifar100_template = [
    lambda c: f'a photo of a {c}.',
    lambda c: f'a blurry photo of a {c}.',
    lambda c: f'a black and white photo of a {c}.',
    lambda c: f'a low contrast photo of a {c}.',
    lambda c: f'a high contrast photo of a {c}.',
    lambda c: f'a bad photo of a {c}.',
    lambda c: f'a good photo of a {c}.',
    lambda c: f'a photo of a small {c}.',
    lambda c: f'a photo of a big {c}.',
    lambda c: f'a photo of the {c}.',
    lambda c: f'a blurry photo of the {c}.',
    lambda c: f'a black and white photo of the {c}.',
    lambda c: f'a low contrast photo of the {c}.',
    lambda c: f'a high contrast photo of the {c}.',
    lambda c: f'a bad photo of the {c}.',
    lambda c: f'a good photo of the {c}.',
    lambda c: f'a photo of the small {c}.',
    lambda c: f'a photo of the big {c}.',
]

dtd_template = [
    lambda c: f'a photo of a {c} texture.',
    lambda c: f'a photo of a {c} pattern.',
    lambda c: f'a photo of a {c} thing.',
    lambda c: f'a photo of a {c} object.',
    lambda c: f'a photo of the {c} texture.',
    lambda c: f'a photo of the {c} pattern.',
    lambda c: f'a photo of the {c} thing.',
    lambda c: f'a photo of the {c} object.',
]

eurosat_template = [
    lambda c: f'a centered satellite photo of {c}.',
    lambda c: f'a centered satellite photo of a {c}.',
    lambda c: f'a centered satellite photo of the {c}.',
]

food101_template = [
    lambda c: f'a photo of a delicious {c}.',
    lambda c: f'a photo of a freshly cooked {c}.',
    lambda c: f'a photo of a {c} on a plate.',
    lambda c: f'I love eating {c}!',
    lambda c: f'a photo of a {c} served at a restaurant.',
    lambda c: f'a close-up photo of a {c}.',
    lambda c: f'a photo of a traditional {c}.',
    lambda c: f'a photo of an organic {c}.',
    lambda c: f'a photo of a homemade {c}.',
    lambda c: f'a photo of a gourmet {c}.',
    lambda c: f'a photo of a {c} with garnish.',
    lambda c: f'a photo of a sliced {c}.',
    lambda c: f'a photo of a healthy {c}.',
    lambda c: f'a photo of a spicy {c}.',
    lambda c: f'a photo of a sweet {c}.',
    lambda c: f'a photo of a savory {c}.',
    lambda c: f'a photo of a hot {c}.',
    lambda c: f'a photo of a cold {c}.',
    lambda c: f'a photo of a perfect {c} for breakfast.',
    lambda c: f'a photo of a perfect {c} for lunch.',
    lambda c: f'a photo of a perfect {c} for dinner.',
    lambda c: f'a photo of a street food {c}.',
    lambda c: f'a photo of a {c} ready to eat.',
    lambda c: f'a photo of a {c} being prepared.',
]

gtsrb_template = [
    lambda c: f'a zoomed in photo of a "{c}" traffic sign.',
    lambda c: f'a centered photo of a "{c}" traffic sign.',
    lambda c: f'a close up photo of a "{c}" traffic sign.',
]

mnist_template = [
    lambda c: f'a photo of the number {c}.',
    lambda c: f'a hand-drawn digit {c}.',
    lambda c: f'a sketch of the number {c}.',
    lambda c: f'an image of the digit {c}.',
    lambda c: f'a black and white photo of the number {c}.',
    lambda c: f'a clear image of the number {c}.',
    lambda c: f'a blurry photo of the number {c}.',
    lambda c: f'an artistic rendering of the digit {c}.',
    lambda c: f'a digit {c} drawn with a pen.',
    lambda c: f'a digit {c} drawn with a pencil.',
    lambda c: f'a bold number {c}.',
    lambda c: f'a faded number {c}.',
    lambda c: f'a number {c} on a white background.',
    lambda c: f'a number {c} on a black background.',
    lambda c: f'a small digit {c}.',
    lambda c: f'a large digit {c}.',
]

imagenet_template = [
    lambda c: f'a bad photo of a {c}.',
    lambda c: f'a photo of many {c}.',
    lambda c: f'a sculpture of a {c}.',
    lambda c: f'a photo of the hard to see {c}.',
    lambda c: f'a low resolution photo of the {c}.',
    lambda c: f'a rendering of a {c}.',
    lambda c: f'graffiti of a {c}.',
    lambda c: f'a bad photo of the {c}.',
    lambda c: f'a cropped photo of the {c}.',
    lambda c: f'a tattoo of a {c}.',
    lambda c: f'the embroidered {c}.',
    lambda c: f'a photo of a hard to see {c}.',
    lambda c: f'a bright photo of a {c}.',
    lambda c: f'a photo of a clean {c}.',
    lambda c: f'a photo of a dirty {c}.',
    lambda c: f'a dark photo of the {c}.',
    lambda c: f'a drawing of a {c}.',
    lambda c: f'a photo of my {c}.',
    lambda c: f'the plastic {c}.',
    lambda c: f'a photo of the cool {c}.',
    lambda c: f'a close-up photo of a {c}.',
    lambda c: f'a black and white photo of the {c}.',
    lambda c: f'a painting of the {c}.',
    lambda c: f'a painting of a {c}.',
    lambda c: f'a pixelated photo of the {c}.',
    lambda c: f'a sculpture of the {c}.',
    lambda c: f'a bright photo of the {c}.',
    lambda c: f'a cropped photo of a {c}.',
    lambda c: f'a plastic {c}.',
    lambda c: f'a photo of the dirty {c}.',
    lambda c: f'a jpeg corrupted photo of a {c}.',
    lambda c: f'a blurry photo of the {c}.',
    lambda c: f'a photo of the {c}.',
    lambda c: f'a good photo of the {c}.',
    lambda c: f'a rendering of the {c}.',
    lambda c: f'a {c} in a video game.',
    lambda c: f'a photo of one {c}.',
    lambda c: f'a doodle of a {c}.',
    lambda c: f'a close-up photo of the {c}.',
    lambda c: f'a photo of a {c}.',
    lambda c: f'the origami {c}.',
    lambda c: f'the {c} in a video game.',
    lambda c: f'a sketch of a {c}.',
    lambda c: f'a doodle of the {c}.',
    lambda c: f'a origami {c}.',
    lambda c: f'a low resolution photo of a {c}.',
    lambda c: f'the toy {c}.',
    lambda c: f'a rendition of the {c}.',
    lambda c: f'a photo of the clean {c}.',
    lambda c: f'a photo of a large {c}.',
    lambda c: f'a rendition of a {c}.',
    lambda c: f'a photo of a nice {c}.',
    lambda c: f'a photo of a weird {c}.',
    lambda c: f'a blurry photo of a {c}.',
    lambda c: f'a cartoon {c}.',
    lambda c: f'art of a {c}.',
    lambda c: f'a sketch of the {c}.',
    lambda c: f'a embroidered {c}.',
    lambda c: f'a pixelated photo of a {c}.',
    lambda c: f'itap of the {c}.',
    lambda c: f'a jpeg corrupted photo of the {c}.',
    lambda c: f'a good photo of a {c}.',
    lambda c: f'a plushie {c}.',
    lambda c: f'a photo of the nice {c}.',
    lambda c: f'a photo of the small {c}.',
    lambda c: f'a photo of the weird {c}.',
    lambda c: f'the cartoon {c}.',
    lambda c: f'art of the {c}.',
    lambda c: f'a drawing of the {c}.',
    lambda c: f'a photo of the large {c}.',
    lambda c: f'a black and white photo of a {c}.',
    lambda c: f'the plushie {c}.',
    lambda c: f'a dark photo of a {c}.',
    lambda c: f'itap of a {c}.',
    lambda c: f'graffiti of the {c}.',
    lambda c: f'a toy {c}.',
    lambda c: f'itap of my {c}.',
    lambda c: f'a photo of a cool {c}.',
    lambda c: f'a photo of a small {c}.',
    lambda c: f'a tattoo of the {c}.',
]

resisc45_template = [
    lambda c: f'satellite imagery of {c}.',
    lambda c: f'aerial imagery of {c}.',
    lambda c: f'satellite photo of {c}.',
    lambda c: f'aerial photo of {c}.',
    lambda c: f'satellite view of {c}.',
    lambda c: f'aerial view of {c}.',
    lambda c: f'satellite imagery of a {c}.',
    lambda c: f'aerial imagery of a {c}.',
    lambda c: f'satellite photo of a {c}.',
    lambda c: f'aerial photo of a {c}.',
    lambda c: f'satellite view of a {c}.',
    lambda c: f'aerial view of a {c}.',
    lambda c: f'satellite imagery of the {c}.',
    lambda c: f'aerial imagery of the {c}.',
    lambda c: f'satellite photo of the {c}.',
    lambda c: f'aerial photo of the {c}.',
    lambda c: f'satellite view of the {c}.',
    lambda c: f'aerial view of the {c}.',
]

stl10_template = [
    lambda c: f'a photo of a {c}.',
    lambda c: f'a photo of the {c}.',
]

sun397_template = [
    lambda c: f'a photo of a {c}.',
    lambda c: f'a photo of the {c}.',
]

svhn_template = [
    lambda c: f'a photo of the number: "{c}".',
]

flowers_template = [
    lambda c: f'a photo of a beautiful {c}.',
    lambda c: f'a close-up image of a {c}.',
    lambda c: f'a photo of a blooming {c}.',
    lambda c: f'an image of a {c} in a garden.',
    lambda c: f'a photo of a colorful {c}.',
    lambda c: f'a detailed photo of a {c} petal.',
    lambda c: f'a photo of a {c} in the wild.',
    lambda c: f'a scenic shot of a {c} in natural light.',
    lambda c: f'a photo of a freshly opened {c}.',
    lambda c: f'a macro photo of a {c}.',
    lambda c: f'a vibrant {c} standing out in its natural habitat.',
    lambda c: f'a serene photo of a {c} by the water.',
    lambda c: f'an artistic shot of a {c}.',
    lambda c: f'a photo of a {c} under the sun.',
    lambda c: f'a photo of a {c} with dew on its petals.',
    lambda c: f'an evening shot of a {c}.',
    lambda c: f'a photo of a rare {c}.',
    lambda c: f'a black and white photo of a {c}.',
    lambda c: f'a photo of a {c} surrounded by foliage.',
    lambda c: f'a bright {c} on a cloudy day.',
    lambda c: f'a photo of a wilting {c}.',
    lambda c: f'a photo of a young {c} bud.',
    lambda c: f'a photo of a {c} in full bloom.'
]

imagenette_template = cifar10_template

imagewoof_template = cifar10_template

fashionmnist_template = [
    lambda c: f'a photo of a {c}.',
    lambda c: f'a photo of the {c}.',
]

real_template = [
    lambda c: f'a photo of a {c} in real life.',
    lambda c: f'a realistic image of a {c}.',
    lambda c: f'an everyday photo of a {c}.',
    lambda c: f'a {c} as seen in the natural environment.',
    lambda c: f'a candid shot of a {c}.',
    lambda c: f'a {c} that you would see on the street.',
]

sketch_template = [
    lambda c: f'a sketch of a {c}.',
    lambda c: f'a pencil drawing of a {c}.',
    lambda c: f'a hand-drawn {c}.',
    lambda c: f'a simple sketch of a {c}.',
    lambda c: f'an artistic sketch of a {c}.',
    lambda c: f'a black and white sketch of a {c}.',
]

painting_template = [
    lambda c: f'an oil painting of a {c}.',
    lambda c: f'a painting depicting a {c}.',
    lambda c: f'an artistic rendering of a {c} in paint.',
    lambda c: f'a classical painting of a {c}.',
    lambda c: f'a {c} as seen in renaissance art.',
    lambda c: f'a detailed painting of a {c}.',
]

clipart_template = [
    lambda c: f'a clip art of a {c}.',
    lambda c: f'a cartoonish {c}.',
    lambda c: f'a stylized image of a {c}.',
    lambda c: f'a graphic representation of a {c}.',
    lambda c: f'a colorful, simple portrayal of a {c}.',
    lambda c: f'an illustration of a {c}.',
]

infograph_template = [
    lambda c: f'an infographic of a {c}.',
    lambda c: f'an informative illustration of a {c}.',
    lambda c: f'a data-driven depiction of a {c}.',
    lambda c: f'a visual statistic about a {c}.',
    lambda c: f'a diagram featuring a {c}.',
    lambda c: f'an illustrated fact about a {c}.',
]

quickdraw_template = [
    lambda c: f'a quick drawing of a {c}.',
    lambda c: f'a hastily sketched {c}.',
    lambda c: f'a simple line drawing of a {c}.',
    lambda c: f'a quick scribble representing a {c}.',
    lambda c: f'a minimalistic drawing of a {c}.',
    lambda c: f'a rough outline of a {c}.',
]

dataset_to_template = {
    'Cars': cars_template,
    'CIFAR10': cifar10_template,
    'CIFAR100': cifar100_template,
    'DTD': dtd_template,
    'EuroSAT': eurosat_template,
    'Food101': food101_template,
    'GTSRB': gtsrb_template,
    'MNIST': mnist_template,
    'ImageNet': imagenet_template,
    'RESISC45': resisc45_template,
    'STL10': stl10_template,
    'SUN397': sun397_template,
    'SVHN': svhn_template,
    'Flowers102': flowers_template,
    'Imagenette': imagenette_template,
    'Imagewoof': imagewoof_template,
    'FashionMNIST': fashionmnist_template,
    'Clipart': clipart_template,
    'Painting': painting_template,
    'Quickdraw': quickdraw_template,
    'Real': real_template,
    'Sketch': sketch_template,
    'Infograph': infograph_template,
}


def get_templates(dataset_name):
    if dataset_name.endswith('Val'):
        return get_templates(dataset_name.replace('Val', ''))
    assert dataset_name in dataset_to_template, f'Unsupported dataset: {dataset_name}'
    return dataset_to_template[dataset_name]