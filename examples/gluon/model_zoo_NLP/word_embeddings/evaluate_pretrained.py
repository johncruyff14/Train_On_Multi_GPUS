# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=logging-too-many-args
"""Evaluation of pre-trained word embeddings
============================================

This example shows how to load and perform intrinsic evaluation of word
embeddings using a variety of datasets all part of the Gluon NLP Toolkit.

"""

import argparse
import logging
import os
import sys

import mxnet as mx

import evaluation
import gluonnlp as nlp
import utils

nlp.utils.check_version('0.7.0')

def get_args():
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(
        description='Word embedding evaluation with Gluon.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Embeddings arguments
    group = parser.add_argument_group('Embedding arguments')
    group.add_argument('--embedding-path', type=str,
                       help='Path to a .vec in Word2Vec text foramt or '
                       '.bin binary fastText model file. ')
    group.add_argument('--embedding-name', type=str,
                       help=('Name of embedding type to load. '
                             'Valid entries: {}'.format(
                                 ', '.join(
                                     nlp.embedding.list_sources().keys()))))
    group.add_argument('--embedding-source', type=str,
                       help=('Source from which to initialize the embedding.'
                             'Pass --list-embedding-sources to get a list of '
                             'valid sources for a given --embedding-name.'))
    group.add_argument(
        '--fasttext-load-ngrams',
        action='store_true',
        help=('Specify load_ngrams=True '
              'when loading pretrained fastText embedding.'))
    group.add_argument(
        '--analogy-max-vocab-size', type=int, default=None,
        help=('Only retain the X first tokens from the pre-trained embedding. '
              'The tokens are ordered by decreasing frequency.'
              'As the analogy task takes the whole vocabulary into account, '
              'removing very infrequent words improves performance.'))
    group.add_argument('--list-embedding-sources', action='store_true')

    # Computation options
    group = parser.add_argument_group('Computation arguments')
    group.add_argument('--batch-size', type=int, default=1024,
                       help='Batch size to use on analogy task. '
                       'Decrease batch size if evaluation crashes.')
    group.add_argument('--gpu', type=int,
                       help=('Number (index) of GPU to run on, e.g. 0. '
                             'If not specified, uses CPU.'))
    group.add_argument('--no-hybridize', action='store_true',
                       help='Disable hybridization of gluon HybridBlocks.')

    # Logging
    group = parser.add_argument_group('Logging arguments')
    group.add_argument('--logdir', type=str, default='logs',
                       help='Directory to store logs.')

    # Evaluation options
    evaluation.add_parameters(parser)

    args = parser.parse_args()

    validate_args(args)
    evaluation.validate_args(args)

    return args


def validate_args(args):
    """Validate provided arguments and act on --help."""
    if args.list_embedding_sources:
        print('Listing all sources for {} embeddings.'.format(
            args.embedding_name))
        print('Specify --embedding-name if you wish to '
              'list sources of other embeddings')
        print('')
        if args.embedding_name not in nlp.embedding.list_sources().keys():
            print('Invalid embedding name.')
            print('Only {} are supported.'.format(', '.join(
                nlp.embedding.list_sources().keys())))
            sys.exit(1)
        print(' '.join(nlp.embedding.list_sources()[args.embedding_name]))
        sys.exit(0)

    if not (args.embedding_path or args.embedding_name):
        print('You must specify either --embedding-path or --embedding-name ')
        print('Use --embedding-path to load and evaluate '
              'word embeddings from a Word2Vec text format '
              'or fastText binary format file')
        print('Use --embedding-name or to download one of '
              'the pre-trained embedding files included in GluonNLP.')
        sys.exit(1)

    if args.embedding_name and not args.embedding_source:
        print('Please also specify --embedding-source'
              ' to select the version of the pre-trained embedding. '
              'Use --list-embedding-sources to see all available sources')
        sys.exit(1)

    print(args)


def load_embedding_from_path(args):
    """Load a TokenEmbedding."""
    if args.embedding_path.endswith('.bin'):
        with utils.print_time('load fastText model.'):
            model = \
                nlp.model.train.FasttextEmbeddingModel.load_fasttext_format(
                    args.embedding_path)
        idx_to_token = sorted(model._token_to_idx, key=model._token_to_idx.get)

        # Analogy task is open-vocabulary, so must keep all known words.
        # But if not evaluating analogy, no need to precompute now as all
        # words for closed vocabulary task can be obtained via the unknown
        # lookup
        if not args.analogy_datasets:
            # TODO(leezu): use shape (0, model.weight.shape[1]) once np shape
            # is supported by TokenEmbedding
            idx_to_token = ['<unk>']
            idx_to_vec = mx.nd.zeros((1,  model.weight.shape[1]))
        else:
            if args.analogy_max_vocab_size:
                idx_to_token = idx_to_token[:args.analogy_max_vocab_size]
            with utils.print_time('compute vectors for {} known '
                                  'words.'.format(len(idx_to_token))):
                idx_to_vec = model[idx_to_token]

        embedding = nlp.embedding.TokenEmbedding(
            unknown_token=None, idx_to_token=idx_to_token,
            idx_to_vec=idx_to_vec, unknown_lookup=model)
    else:
        embedding = nlp.embedding.TokenEmbedding.from_file(args.embedding_path)

    return embedding


def load_embedding_from_gluonnlp(args):
    if args.embedding_name.lower() == 'fasttext':
        token_embedding = nlp.embedding.create(
            args.embedding_name,
            source=args.embedding_source,
            load_ngrams=args.fasttext_load_ngrams)
    else:
        token_embedding = nlp.embedding.create(
            args.embedding_name, source=args.embedding_source)
    return token_embedding


def enforce_max_size(token_embedding, size):
    assert token_embedding.idx_to_vec is not None
    if size and len(token_embedding.idx_to_token) > size:
        assert size > 0
        size = size + 1 if token_embedding.unknown_token is not None else size
        token_embedding = nlp.embedding.TokenEmbedding(
            unknown_token=token_embedding.unknown_token,
            idx_to_token=token_embedding._idx_to_token[:size],
            idx_to_vec=token_embedding._idx_to_vec[:size],
            unknown_lookup=token_embedding.unknown_lookup)
    return token_embedding

def run_test(args_):
    if args_.similarity_datasets:
        with utils.print_time('find relevant tokens for similarity'):
            tokens = evaluation.get_similarity_task_tokens(args_)
        vocab = nlp.Vocab(nlp.data.count_tokens(tokens),
                      unknown_token=token_embedding_.unknown_token,
                      padding_token=None, bos_token=None, eos_token=None)
        with utils.print_time('set {} embeddings'.format(len(tokens))):
            vocab.set_embedding(token_embedding_)
        evaluation.evaluate_similarity(
            args_, vocab.embedding, ctx, logfile=os.path.join(
            args_.logdir, 'similarity{}.tsv'.format(name)))
    if args_.analogy_datasets:
        with utils.print_time('extend open vocabulary with '
                          'OOV tokens for analogy'):
            tokens = evaluation.get_analogy_task_tokens(args_)
        if token_embedding_.unknown_token is not None:
            tokens.update(token_embedding_.idx_to_token[1:])
        else:
            tokens.update(token_embedding_.idx_to_token)
        vocab = nlp.Vocab(nlp.data.count_tokens(tokens),
                      unknown_token=token_embedding_.unknown_token,
                      padding_token=None, bos_token=None, eos_token=None)
        with utils.print_time('set {} embeddings'.format(len(tokens))):
            vocab.set_embedding(token_embedding_)
        evaluation.evaluate_analogy(
            args_, vocab.embedding, ctx, logfile=os.path.join(
            args_.logdir, 'analogy{}.tsv'.format(name)))

if __name__ == '__main__':
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    args_ = get_args()
    ctx = utils.get_context(args_)[0]
    if not os.path.isdir(args_.logdir):
        os.makedirs(args_.logdir)

    # Load pre-trained embeddings
    if not args_.embedding_path:
        token_embedding_ = load_embedding_from_gluonnlp(args_)
        name = '-' + args_.embedding_name + '-' + args_.embedding_source
    else:
        token_embedding_ = load_embedding_from_path(args_)
        name = ''

    token_embedding_ = enforce_max_size(
        token_embedding_, args_.analogy_max_vocab_size)
    if args_.fasttext_load_ngrams:
        assert token_embedding_.unknown_lookup is not None
    known_tokens = set(token_embedding_.idx_to_token)

    catch_exception = os.getenv('CATCH_EXEPTION')
    if catch_exception and int(catch_exception) != 0:
        try:
            run_test(args_)
        except:
            print("===>>> Evaluation failed!!!\n")
    else:
        run_test(args_)

