#! /usr/bin/env python
import sys
sys.path.append('/Users/danilonunes/workspace/phd/fast-rcnn/caffe-fast-rcnn/python')
sys.path.append('/Users/danilonunes/workspace/phd/fast-rcnn/lib')
import caffe
import click
import cPickle as pickle
from datasets.factory import get_imdb
from rcc8.rcc import RCC8


@click.group('common_tasks')
def main():
    pass


@main.command()
@click.option('--dataset', '-d', default='voc_2007_val', help="Dataset")
@click.option('--output_file', '-f', default='relations.pkl', help="Output file")
def test(dataset, output_file):
    """
    Create a pickle file from relations according to annotations in a dataset (VOC 2007)
    """
    imdb = get_imdb(dataset)

    db_relations = {}
    rcc_detector = RCC8()

    for n, image_index in enumerate(imdb.image_index):
        if n % 10 == 0:
            print u'Processed: {} / {}'.format(n, len(imdb.image_index))

        impath = imdb.image_path_from_index(image_index)

        # read image
        img = caffe.io.load_image(impath)

        # read annotations
        annotations = imdb._load_pascal_annotation(image_index)

        # compute RCC8 relation
        hist, relations, pairs = rcc_detector.get_relations(img,
                                                            annotations=annotations['gt_classes'],
                                                            objects=annotations['boxes'])
        db_relations[image_index] = (hist, relations, pairs)

    if db_relations:
        with open(output_file, 'wb') as handle:
            pickle.dump(db_relations, handle)
    else:
        print u'Relations not found!'


if __name__ == '__main__':
    main()
