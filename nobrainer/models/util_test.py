# -*- coding: utf-8 -*-
"""Tests for `nobrainer.models.util`"""

import pytest
import tensorflow as tf

import nobrainer

from nobrainer.models.util import check_optimizer_for_training
from nobrainer.models.util import check_required_params
from nobrainer.models.util import get_estimator
from nobrainer.models.util import get_items_not_in_iterable


def test_get_estimator():
    est_obj = get_estimator('highres3dnet')
    assert est_obj == nobrainer.models.highres3dnet.HighRes3DNet

    _model = nobrainer.models.HighRes3DNet
    assert get_estimator(_model) is _model

    with pytest.raises(ValueError):
        get_estimator('fakemodel')


def test_get_items_not_in_iterable():
    items = {'foo', 'bar'}
    iterable = ['foo', 'baz']
    assert get_items_not_in_iterable(items=items, iterable=iterable) == {'bar'}

    items = {'foo', 'bar'}
    iterable = ['foo', 'bar', 'baz']
    assert not get_items_not_in_iterable(items=items, iterable=iterable)


def test_check_required_params():
    required = {'foo', 'bar'}
    params = {'foo': 0, 'baz': 1}

    with pytest.raises(ValueError):
        check_required_params(required_keys=required, params=params)

    params['bar'] = 2
    check_required_params(required_keys=required, params=params)


def set_default_params():
    params = {'foo': 0, 'bar': 1}
    defaults = {'foo': 10, 'bar': 20, 'baz': 30}
    set_default_params(defaults=defaults, params=params)

    assert params['foo'] == 0
    assert params['bar'] == 1
    assert params['baz'] == 30


def test_check_optimizer_for_training():
    with pytest.raises(ValueError):
        check_optimizer_for_training(
            optimizer=None, mode=tf.estimator.ModeKeys.TRAIN)

    check_optimizer_for_training(
        optimizer='Adam', mode=tf.estimator.ModeKeys.TRAIN)
    check_optimizer_for_training(
        optimizer=None, mode=tf.estimator.ModeKeys.EVAL)
    check_optimizer_for_training(
        optimizer=None, mode=tf.estimator.ModeKeys.PREDICT)
