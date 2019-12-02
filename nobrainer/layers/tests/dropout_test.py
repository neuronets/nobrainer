"""Tests for `nobrainer.layers.dropout`.

The reference values come from functions implemented in
https://github.com/patrick-mcclure/nobrainer/blob/92942a31794c4306b6cb406c8a89539d9b192e69/nobrainer/models/bayesian_dropout.py
using TensorFlow version 1.13.1.
"""

from numpy.testing import assert_allclose
import tensorflow as tf

from nobrainer.layers import dropout as dpl

a = tf.constant(
    [[-0.3199792 ,  0.27815193,  1.185772  , -0.6832058 , -0.04553711,
         0.9464889 , -0.7914983 , -0.88987124, -2.0239818 ,  0.38825727],
       [-1.5746268 ,  1.5496527 , -0.68980837, -0.3384262 , -1.5759798 ,
         0.13767439,  0.94831973,  0.08519623, -0.3011496 ,  1.694583  ],
       [-1.5999061 , -1.6850958 , -2.370195  , -1.5993102 ,  0.07427412,
        -0.7281705 , -0.93446416, -0.03413244,  1.1198186 ,  0.565427  ],
       [-0.14894725, -0.97795886, -0.9567534 ,  0.23645365, -1.0828192 ,
         0.20657389,  0.30088323,  1.1067911 , -0.6242912 ,  1.6529874 ],
       [ 0.48662138, -0.47715223, -0.14808841,  1.7914195 , -0.62888914,
         0.19279727, -1.4153436 , -0.32482007, -2.0423434 , -0.11563149],
       [ 0.56918573, -0.6642863 , -1.6188092 ,  0.14785443,  1.0411389 ,
         0.84223944, -0.25680584,  1.4933782 ,  1.8803914 ,  0.9905541 ],
       [-1.5538213 , -0.25285706,  1.2355319 ,  1.3271157 , -0.97071636,
        -0.61771345, -0.246103  , -0.76273334, -0.39202142, -1.7139134 ],
       [-0.36039367, -1.6153171 , -0.23746784, -1.4615673 , -1.5481923 ,
        -0.9850169 ,  0.15943421,  1.1271275 ,  0.5603631 , -0.00874051],
       [ 0.55738926, -0.0810589 , -0.31762588,  0.18993841, -1.2941253 ,
         0.24133618,  0.46214277, -1.8823439 ,  1.2868234 ,  2.0315878 ],
       [-0.8111379 ,  1.1777685 ,  1.5922145 , -0.5105937 , -2.0080373 ,
         0.02225626,  0.68188673, -0.84452343,  2.0827887 ,  1.841939  ]],
                dtype=tf.float32)


def test_bernoulli_dropout():
    reference_rate20_mc_scale = tf.constant(
    [[-0.399974  ,  0.3476899 ,  1.4822149 , -0.85400724, -0.05692139,
            1.1831111 , -0.98937285, -1.112339  , -2.5299773 ,  0.48532158],
        [-0.        ,  1.9370658 , -0.86226046, -0.42303276, -1.9699748 ,
            0.17209299,  1.1853997 ,  0.        , -0.376437  ,  2.1182287 ],
        [-1.9998826 , -2.1063697 , -0.        , -1.9991376 ,  0.09284265,
        -0.9102131 , -1.1680802 , -0.04266555,  1.3997732 ,  0.        ],
        [-0.18618406, -0.        , -1.1959417 ,  0.29556707, -1.353524  ,
            0.        ,  0.37610403,  1.3834889 , -0.        ,  2.066234  ],
        [ 0.6082767 , -0.59644026, -0.18511051,  2.2392743 , -0.        ,
            0.24099658, -1.7691796 , -0.40602508, -0.        , -0.        ],
        [ 0.71148217, -0.8303579 , -2.0235114 ,  0.18481804,  0.        ,
            1.0527992 , -0.32100728,  0.        ,  0.        ,  1.2381926 ],
        [-1.9422766 , -0.31607133,  0.        ,  0.        , -1.2133955 ,
        -0.        , -0.30762875, -0.        , -0.49002677, -2.1423917 ],
        [-0.        , -2.0191464 , -0.2968348 , -1.8269591 , -1.9352403 ,
        -1.231271  ,  0.19929276,  1.4089094 ,  0.7004539 , -0.01092564],
        [ 0.6967366 , -0.10132362, -0.39703235,  0.23742302, -1.6176566 ,
            0.30167022,  0.        , -2.3529298 ,  1.6085292 ,  2.5394847 ],
        [-1.0139223 ,  0.        ,  1.9902681 , -0.6382421 , -2.5100467 ,
            0.02782033,  0.8523584 , -1.0556543 ,  2.6034858 ,  2.3024237 ]],
                                        dtype=tf.float32)

    reference_rate20_mc_noscale = tf.constant(
    [[-0.3199792 ,  0.27815193,  1.185772  , -0.6832058 , -0.04553711,
            0.94648886, -0.7914983 , -0.88987124, -2.0239818 ,  0.38825727],
        [-0.        ,  1.5496527 , -0.68980837, -0.3384262 , -1.5759798 ,
            0.13767439,  0.94831973,  0.        , -0.3011496 ,  1.6945829 ],
        [-1.5999061 , -1.6850958 , -0.        , -1.5993102 ,  0.07427412,
        -0.7281705 , -0.93446416, -0.03413244,  1.1198186 ,  0.        ],
        [-0.14894725, -0.        , -0.9567534 ,  0.23645365, -1.0828192 ,
            0.        ,  0.30088323,  1.1067911 , -0.        ,  1.6529874 ],
        [ 0.48662138, -0.4771522 , -0.14808841,  1.7914194 , -0.        ,
            0.19279727, -1.4153436 , -0.32482007, -0.        , -0.        ],
        [ 0.56918573, -0.6642863 , -1.6188091 ,  0.14785443,  0.        ,
            0.8422394 , -0.25680584,  0.        ,  0.        ,  0.99055403],
        [-1.5538213 , -0.25285706,  0.        ,  0.        , -0.9707164 ,
        -0.        , -0.246103  , -0.        , -0.39202142, -1.7139133 ],
        [-0.        , -1.6153172 , -0.23746784, -1.4615673 , -1.5481923 ,
        -0.9850168 ,  0.15943421,  1.1271275 ,  0.5603631 , -0.00874051],
        [ 0.55738926, -0.0810589 , -0.31762588,  0.18993841, -1.2941253 ,
            0.24133618,  0.        , -1.8823439 ,  1.2868234 ,  2.0315878 ],
        [-0.81113786,  0.        ,  1.5922145 , -0.5105937 , -2.0080373 ,
            0.02225626,  0.68188673, -0.84452343,  2.0827887 ,  1.841939  ]],
    dtype=tf.float32
    )

    reference_rate20_nomc_scale = tf.constant(
    [[-0.3199792 ,  0.27815193,  1.185772  , -0.6832058 , -0.04553711,
            0.9464889 , -0.7914983 , -0.88987124, -2.0239818 ,  0.38825727],
        [-1.5746268 ,  1.5496527 , -0.68980837, -0.3384262 , -1.5759798 ,
            0.13767439,  0.94831973,  0.08519623, -0.3011496 ,  1.694583  ],
        [-1.5999061 , -1.6850958 , -2.370195  , -1.5993102 ,  0.07427412,
        -0.7281705 , -0.93446416, -0.03413244,  1.1198186 ,  0.565427  ],
        [-0.14894725, -0.97795886, -0.9567534 ,  0.23645365, -1.0828192 ,
            0.20657389,  0.30088323,  1.1067911 , -0.6242912 ,  1.6529874 ],
        [ 0.48662138, -0.47715223, -0.14808841,  1.7914195 , -0.62888914,
            0.19279727, -1.4153436 , -0.32482007, -2.0423434 , -0.11563149],
        [ 0.56918573, -0.6642863 , -1.6188092 ,  0.14785443,  1.0411389 ,
            0.84223944, -0.25680584,  1.4933782 ,  1.8803914 ,  0.9905541 ],
        [-1.5538213 , -0.25285706,  1.2355319 ,  1.3271157 , -0.97071636,
        -0.61771345, -0.246103  , -0.76273334, -0.39202142, -1.7139134 ],
        [-0.36039367, -1.6153171 , -0.23746784, -1.4615673 , -1.5481923 ,
        -0.9850169 ,  0.15943421,  1.1271275 ,  0.5603631 , -0.00874051],
        [ 0.55738926, -0.0810589 , -0.31762588,  0.18993841, -1.2941253 ,
            0.24133618,  0.46214277, -1.8823439 ,  1.2868234 ,  2.0315878 ],
        [-0.8111379 ,  1.1777685 ,  1.5922145 , -0.5105937 , -2.0080373 ,
            0.02225626,  0.68188673, -0.84452343,  2.0827887 ,  1.841939  ]],
                                            dtype=tf.float32)

    reference_rate20_nomc_noscale = tf.constant(
    [[-0.25598335,  0.22252154,  0.9486176 , -0.54656464, -0.03642969,
            0.7571911 , -0.6331987 , -0.711897  , -1.6191854 ,  0.31060582],
        [-1.2597015 ,  1.2397221 , -0.5518467 , -0.27074096, -1.2607839 ,
            0.11013951,  0.7586558 ,  0.06815698, -0.2409197 ,  1.3556665 ],
        [-1.2799249 , -1.3480767 , -1.896156  , -1.2794482 ,  0.0594193 ,
        -0.5825364 , -0.74757135, -0.02730595,  0.8958549 ,  0.45234162],
        [-0.11915781, -0.7823671 , -0.76540273,  0.18916292, -0.8662554 ,
            0.16525911,  0.2407066 ,  0.8854329 , -0.49943295,  1.32239   ],
        [ 0.3892971 , -0.3817218 , -0.11847073,  1.4331356 , -0.5031113 ,
            0.15423782, -1.132275  , -0.25985608, -1.6338748 , -0.09250519],
        [ 0.45534858, -0.53142905, -1.2950474 ,  0.11828355,  0.83291113,
            0.6737916 , -0.20544468,  1.1947025 ,  1.5043131 ,  0.7924433 ],
        [-1.2430571 , -0.20228565,  0.98842555,  1.0616926 , -0.7765731 ,
        -0.49417076, -0.19688241, -0.6101867 , -0.31361714, -1.3711308 ],
        [-0.28831494, -1.2922537 , -0.18997428, -1.1692538 , -1.2385539 ,
        -0.7880135 ,  0.12754737,  0.90170205,  0.4482905 , -0.00699241],
        [ 0.4459114 , -0.06484712, -0.2541007 ,  0.15195073, -1.0353003 ,
            0.19306895,  0.36971423, -1.5058751 ,  1.0294588 ,  1.6252702 ],
        [-0.64891034,  0.9422148 ,  1.2737716 , -0.40847498, -1.6064299 ,
            0.01780501,  0.5455094 , -0.67561877,  1.666231  ,  1.4735512 ]],
    dtype=tf.float32)

    seed = 42
    kwds = dict(rtol=1e-05, atol=1e-08)

    tf.random.set_seed(seed)
    y_ = dpl.BernoulliDropout(rate=0.2, is_monte_carlo=False, scale_during_training=False, seed=seed)(a)
    assert_allclose(y_, reference_rate20_nomc_noscale, **kwds)

    tf.random.set_seed(seed)
    y_ = dpl.BernoulliDropout(rate=0.2, is_monte_carlo=True, scale_during_training=False, seed=seed)(a)
    assert_allclose(y_, reference_rate20_mc_noscale, **kwds)

    tf.random.set_seed(seed)
    y_ = dpl.BernoulliDropout(rate=0.2, is_monte_carlo=False, scale_during_training=True, seed=seed)(a)
    assert_allclose(y_, reference_rate20_nomc_scale, **kwds)

    tf.random.set_seed(seed)
    y_ = dpl.BernoulliDropout(rate=0.2, is_monte_carlo=True, scale_during_training=True, seed=seed)(a)
    assert_allclose(y_, reference_rate20_mc_scale, **kwds)


def test_concrete_dropout():
    reference_nomc_noexpect = tf.constant(
        [[-0.3199792 ,  0.27815193,  1.185772  , -0.6832058 , -0.04553711,
            0.9464889 , -0.7914983 , -0.88987124, -2.0239818 ,  0.38825727],
        [-1.5746268 ,  1.5496527 , -0.68980837, -0.3384262 , -1.5759798 ,
            0.13767439,  0.94831973,  0.08519623, -0.3011496 ,  1.694583  ],
        [-1.5999061 , -1.6850958 , -2.370195  , -1.5993102 ,  0.07427412,
            -0.7281705 , -0.93446416, -0.03413244,  1.1198186 ,  0.565427  ],
        [-0.14894725, -0.97795886, -0.9567534 ,  0.23645365, -1.0828192 ,
            0.20657389,  0.30088323,  1.1067911 , -0.6242912 ,  1.6529874 ],
        [ 0.48662138, -0.47715223, -0.14808841,  1.7914195 , -0.62888914,
            0.19279727, -1.4153436 , -0.32482007, -2.0423434 , -0.11563149],
        [ 0.56918573, -0.6642863 , -1.6188092 ,  0.14785443,  1.0411389 ,
            0.84223944, -0.25680584,  1.4933782 ,  1.8803914 ,  0.9905541 ],
        [-1.5538213 , -0.25285706,  1.2355319 ,  1.3271157 , -0.97071636,
            -0.61771345, -0.246103  , -0.76273334, -0.39202142, -1.7139134 ],
        [-0.36039367, -1.6153171 , -0.23746784, -1.4615673 , -1.5481923 ,
            -0.9850169 ,  0.15943421,  1.1271275 ,  0.5603631 , -0.00874051],
        [ 0.55738926, -0.0810589 , -0.31762588,  0.18993841, -1.2941253 ,
            0.24133618,  0.46214277, -1.8823439 ,  1.2868234 ,  2.0315878 ],
        [-0.8111379 ,  1.1777685 ,  1.5922145 , -0.5105937 , -2.0080373 ,
            0.02225626,  0.68188673, -0.84452343,  2.0827887 ,  1.841939  ]],
        dtype=tf.float32)

    reference_mc_noexpect = tf.constant(
        [[-0.3199792 ,  0.27815193,  1.185772  , -0.6832058 , -0.04553711,
            0.9464889 , -0.7914983 , -0.88987124, -2.0239818 ,  0.38825727],
        [-0.00000122,  1.5496527 , -0.68980837, -0.3384262 , -1.5759798 ,
            0.13767439,  0.94831973,  0.08518686, -0.3011496 ,  1.694583  ],
        [-1.5999061 , -1.6850958 , -0.        , -1.5993102 ,  0.07427412,
            -0.7281705 , -0.93446416, -0.03413244,  1.1198186 ,  0.04090456],
        [-0.14894725, -0.97795886, -0.9567534 ,  0.23645365, -1.0828192 ,
            0.20582299,  0.30088323,  1.1067911 , -0.        ,  1.6529874 ],
        [ 0.48662138, -0.47715223, -0.14808841,  1.7914195 , -0.        ,
            0.19279727, -1.4153436 , -0.32482007, -2.0423434 , -0.11563107],
        [ 0.56918573, -0.6642863 , -1.6188092 ,  0.14785443,  0.        ,
            0.84223944, -0.25680584,  1.4933782 ,  1.8791399 ,  0.9905541 ],
        [-1.5538213 , -0.25285706,  1.2355319 ,  0.        , -0.97071636,
            -0.00000959, -0.246103  , -0.        , -0.39202142, -1.7139134 ],
        [-0.00000954, -1.6153171 , -0.23746784, -1.4615673 , -1.5481923 ,
            -0.9850169 ,  0.15943421,  1.1271275 ,  0.5603631 , -0.00874051],
        [ 0.55738926, -0.0810589 , -0.31762588,  0.18993841, -1.2941253 ,
            0.24133618,  0.46214277, -1.8823439 ,  1.2868234 ,  2.0315878 ],
        [-0.8111379 ,  1.1776103 ,  1.5922145 , -0.5105937 , -2.0080373 ,
            0.02225626,  0.68188673, -0.84452343,  2.0827887 ,  1.841939  ]],
        dtype=tf.float32
    )

    reference_nomc_expect = tf.constant(
        [[-0.28798127,  0.25033674,  1.0671947 , -0.6148852 , -0.0409834 ,
            0.85184   , -0.71234846, -0.80088407, -1.8215836 ,  0.3494315 ],
        [-1.4171641 ,  1.3946874 , -0.6208275 , -0.30458358, -1.4183818 ,
            0.12390695,  0.85348773,  0.0766766 , -0.27103463,  1.5251247 ],
        [-1.4399154 , -1.5165862 , -2.1331754 , -1.4393791 ,  0.06684671,
            -0.6553534 , -0.8410177 , -0.03071919,  1.0078367 ,  0.5088843 ],
        [-0.13405253, -0.88016295, -0.861078  ,  0.21280828, -0.97453725,
            0.1859165 ,  0.2707949 ,  0.996112  , -0.56186205,  1.4876885 ],
        [ 0.43795922, -0.42943698, -0.13327956,  1.6122775 , -0.5660002 ,
            0.17351754, -1.2738092 , -0.29233804, -1.838109  , -0.10406834],
        [ 0.5122672 , -0.59785765, -1.4569283 ,  0.13306898,  0.93702495,
            0.75801545, -0.23112525,  1.3440403 ,  1.6923522 ,  0.8914987 ],
        [-1.3984392 , -0.22757135,  1.1119787 ,  1.194404  , -0.8736447 ,
            -0.5559421 , -0.2214927 , -0.68645996, -0.35281926, -1.5425221 ],
        [-0.3243543 , -1.4537853 , -0.21372105, -1.3154105 , -1.393373  ,
            -0.8865152 ,  0.14349079,  1.0144148 ,  0.50432676, -0.00786646],
        [ 0.50165033, -0.07295301, -0.28586328,  0.17094457, -1.1647128 ,
            0.21720256,  0.41592848, -1.6941094 ,  1.158141  ,  1.828429  ],
        [-0.7300241 ,  1.0599916 ,  1.4329929 , -0.45953432, -1.8072336 ,
            0.02003063,  0.61369807, -0.76007104,  1.8745098 ,  1.657745  ]],
        dtype=tf.float32
    )

    reference_mc_expect = tf.constant(
        [[-0.3199792 ,  0.27815193,  1.185772  , -0.6832058 , -0.04553711,
            0.9464889 , -0.7914983 , -0.88987124, -2.0239818 ,  0.38825727],
        [-0.00000122,  1.5496527 , -0.68980837, -0.3384262 , -1.5759798 ,
            0.13767439,  0.94831973,  0.08518686, -0.3011496 ,  1.694583  ],
        [-1.5999061 , -1.6850958 , -0.        , -1.5993102 ,  0.07427412,
            -0.7281705 , -0.93446416, -0.03413244,  1.1198186 ,  0.04090456],
        [-0.14894725, -0.97795886, -0.9567534 ,  0.23645365, -1.0828192 ,
            0.20582299,  0.30088323,  1.1067911 , -0.        ,  1.6529874 ],
        [ 0.48662138, -0.47715223, -0.14808841,  1.7914195 , -0.        ,
            0.19279727, -1.4153436 , -0.32482007, -2.0423434 , -0.11563107],
        [ 0.56918573, -0.6642863 , -1.6188092 ,  0.14785443,  0.        ,
            0.84223944, -0.25680584,  1.4933782 ,  1.8791399 ,  0.9905541 ],
        [-1.5538213 , -0.25285706,  1.2355319 ,  0.        , -0.97071636,
            -0.00000959, -0.246103  , -0.        , -0.39202142, -1.7139134 ],
        [-0.00000954, -1.6153171 , -0.23746784, -1.4615673 , -1.5481923 ,
            -0.9850169 ,  0.15943421,  1.1271275 ,  0.5603631 , -0.00874051],
        [ 0.55738926, -0.0810589 , -0.31762588,  0.18993841, -1.2941253 ,
            0.24133618,  0.46214277, -1.8823439 ,  1.2868234 ,  2.0315878 ],
        [-0.8111379 ,  1.1776103 ,  1.5922145 , -0.5105937 , -2.0080373 ,
            0.02225626,  0.68188673, -0.84452343,  2.0827887 ,  1.841939  ]],
        dtype=tf.float32
    )

    seed = 42
    kwds = dict(rtol=1e-05, atol=1e-08)

    tf.random.set_seed(seed)
    y_ = dpl.ConcreteDropout(is_monte_carlo=False, use_expectation=False, seed=seed)(a)
    assert_allclose(y_, reference_nomc_noexpect, **kwds)

    tf.random.set_seed(seed)
    y_ = dpl.ConcreteDropout(is_monte_carlo=True, use_expectation=False, seed=seed)(a)
    assert_allclose(y_, reference_mc_noexpect, **kwds)

    tf.random.set_seed(seed)
    y_ = dpl.ConcreteDropout(is_monte_carlo=False, use_expectation=True, seed=seed)(a)
    assert_allclose(y_, reference_nomc_expect, **kwds)

    tf.random.set_seed(seed)
    y_ = dpl.ConcreteDropout(is_monte_carlo=True, use_expectation=True, seed=seed)(a)
    assert_allclose(y_, reference_mc_expect, **kwds)


def test_gaussian_dropout():
    reference_rate20_mc_scale = tf.constant(
        [[-0.5303459 ,  0.25670424,  1.7261211 , -0.41021588, -0.04306096,
            1.0810643 , -1.0947064 , -0.6135089 , -3.0333736 ,  0.43088648],
        [-0.72443324,  0.00969839, -1.2279863 , -0.30981314, -0.49009138,
            0.11715984,  1.3470882 ,  0.1244951 , -0.22830267,  0.25662357],
        [-1.6761935 , -1.5951506 , -0.25606704, -2.970392  ,  0.03059568,
            -1.1464661 , -1.1449698 , -0.02018523,  1.6528186 ,  0.9587673 ],
        [-0.16076046, -1.0292435 , -1.5045505 ,  0.11255782, -1.1989949 ,
            0.23317099,  0.15103742,  1.4122766 , -0.34741375,  4.4644747 ],
        [ 0.5700574 , -0.37242994, -0.03496783,  1.0465082 , -0.73687136,
            -0.07702249, -0.75437737, -0.25887498, -3.3577132 , -0.19019876],
        [ 0.4627049 , -0.19202873, -1.5101509 ,  0.22365557, -0.65014833,
            1.632061  , -0.36646035,  1.9562665 ,  1.6790464 ,  2.0173438 ],
        [-1.705209  , -0.20624182,  1.435392  ,  2.50725   , -1.2973058 ,
            -0.9860572 , -0.24971803, -1.2850239 , -0.22654453, -1.8393983 ],
        [ 0.03011071, -1.1788267 , -0.07704436, -0.6140916 , -1.3356797 ,
            -0.6224616 ,  0.19039008,  0.9870623 ,  0.33662245, -0.00670144],
        [ 0.87193334, -0.03448428, -0.35466003,  0.11783248, -1.5441091 ,
            0.07059835,  0.05415722, -1.2555441 ,  0.9305809 ,  0.5919759 ],
        [-1.2593663 ,  1.8954549 ,  1.0650359 , -0.7007615 , -0.8861745 ,
            0.03025542,  0.8567357 , -0.5091928 ,  3.3752267 ,  1.0694957 ]],
        dtype=tf.float32
    )

    reference_rate20_mc_noscale = tf.constant(
        [[-0.48827255,  0.26099378,  1.6180512 , -0.4648138 , -0.04355619,
            1.0541493 , -1.0340649 , -0.66878134, -2.8314953 ,  0.4223606 ],
        [-0.8944719 ,  0.3176892 , -1.1203508 , -0.31553575, -0.7072691 ,
            0.12126275,  1.2673346 ,  0.11663534, -0.24287206,  0.54421544],
        [-1.660936  , -1.6131396 , -0.67889255, -2.6961756 ,  0.03933137,
            -1.062807  , -1.1028687 , -0.02297468,  1.5462185 ,  0.88009924],
        [-0.15839782, -1.0189865 , -1.394991  ,  0.13733697, -1.1757598 ,
            0.22785155,  0.1810066 ,  1.3511795 , -0.4027892 ,  3.9021776 ],
        [ 0.5533702 , -0.39337438, -0.05759195,  1.1954905 , -0.71527493,
            -0.02305854, -0.8865706 , -0.272064  , -3.0946393 , -0.17528531],
        [ 0.48400107, -0.28648022, -1.5318826 ,  0.20849535, -0.31189096,
            1.4740968 , -0.34452945,  1.8636888 ,  1.7193154 ,  1.8119859 ],
        [-1.6749315 , -0.21556485,  1.39542   ,  2.2712233 , -1.231988  ,
            -0.91238844, -0.248995  , -1.1805658 , -0.2596399 , -1.8143013 ],
        [-0.04799017, -1.2661248 , -0.10912906, -0.7835867 , -1.3781823 ,
            -0.69497263,  0.18419889,  1.0150753 ,  0.3813706 , -0.00710926],
        [ 0.8090246 , -0.04379921, -0.3472532 ,  0.13225366, -1.4941124 ,
            0.1047459 ,  0.13575432, -1.380904  ,  1.0018294 ,  0.87989825],
        [-1.1697206 ,  1.7519176 ,  1.1704717 , -0.66272795, -1.1105471 ,
            0.02865559,  0.8217659 , -0.5762589 ,  3.116739  ,  1.2239842 ]],
        dtype=tf.float32)

    reference_rate20_nomc_scale = tf.constant(
        [[-0.3199792 ,  0.27815193,  1.185772  , -0.6832058 , -0.04553711,
            0.9464889 , -0.7914983 , -0.88987124, -2.0239818 ,  0.38825727],
        [-1.5746268 ,  1.5496527 , -0.68980837, -0.3384262 , -1.5759798 ,
            0.13767439,  0.94831973,  0.08519623, -0.3011496 ,  1.694583  ],
        [-1.5999061 , -1.6850958 , -2.370195  , -1.5993102 ,  0.07427412,
            -0.7281705 , -0.93446416, -0.03413244,  1.1198186 ,  0.565427  ],
        [-0.14894725, -0.97795886, -0.9567534 ,  0.23645365, -1.0828192 ,
            0.20657389,  0.30088323,  1.1067911 , -0.6242912 ,  1.6529874 ],
        [ 0.48662138, -0.47715223, -0.14808841,  1.7914195 , -0.62888914,
            0.19279727, -1.4153436 , -0.32482007, -2.0423434 , -0.11563149],
        [ 0.56918573, -0.6642863 , -1.6188092 ,  0.14785443,  1.0411389 ,
            0.84223944, -0.25680584,  1.4933782 ,  1.8803914 ,  0.9905541 ],
        [-1.5538213 , -0.25285706,  1.2355319 ,  1.3271157 , -0.97071636,
            -0.61771345, -0.246103  , -0.76273334, -0.39202142, -1.7139134 ],
        [-0.36039367, -1.6153171 , -0.23746784, -1.4615673 , -1.5481923 ,
            -0.9850169 ,  0.15943421,  1.1271275 ,  0.5603631 , -0.00874051],
        [ 0.55738926, -0.0810589 , -0.31762588,  0.18993841, -1.2941253 ,
            0.24133618,  0.46214277, -1.8823439 ,  1.2868234 ,  2.0315878 ],
        [-0.8111379 ,  1.1777685 ,  1.5922145 , -0.5105937 , -2.0080373 ,
            0.02225626,  0.68188673, -0.84452343,  2.0827887 ,  1.841939  ]],
        dtype=tf.float32
    )
    reference_rate20_nomc_noscale = tf.constant(
        [[-0.3199792 ,  0.27815193,  1.185772  , -0.6832058 , -0.04553711,
            0.9464889 , -0.7914983 , -0.88987124, -2.0239818 ,  0.38825727],
        [-1.5746268 ,  1.5496527 , -0.68980837, -0.3384262 , -1.5759798 ,
            0.13767439,  0.94831973,  0.08519623, -0.3011496 ,  1.694583  ],
        [-1.5999061 , -1.6850958 , -2.370195  , -1.5993102 ,  0.07427412,
            -0.7281705 , -0.93446416, -0.03413244,  1.1198186 ,  0.565427  ],
        [-0.14894725, -0.97795886, -0.9567534 ,  0.23645365, -1.0828192 ,
            0.20657389,  0.30088323,  1.1067911 , -0.6242912 ,  1.6529874 ],
        [ 0.48662138, -0.47715223, -0.14808841,  1.7914195 , -0.62888914,
            0.19279727, -1.4153436 , -0.32482007, -2.0423434 , -0.11563149],
        [ 0.56918573, -0.6642863 , -1.6188092 ,  0.14785443,  1.0411389 ,
            0.84223944, -0.25680584,  1.4933782 ,  1.8803914 ,  0.9905541 ],
        [-1.5538213 , -0.25285706,  1.2355319 ,  1.3271157 , -0.97071636,
            -0.61771345, -0.246103  , -0.76273334, -0.39202142, -1.7139134 ],
        [-0.36039367, -1.6153171 , -0.23746784, -1.4615673 , -1.5481923 ,
            -0.9850169 ,  0.15943421,  1.1271275 ,  0.5603631 , -0.00874051],
        [ 0.55738926, -0.0810589 , -0.31762588,  0.18993841, -1.2941253 ,
            0.24133618,  0.46214277, -1.8823439 ,  1.2868234 ,  2.0315878 ],
        [-0.8111379 ,  1.1777685 ,  1.5922145 , -0.5105937 , -2.0080373 ,
            0.02225626,  0.68188673, -0.84452343,  2.0827887 ,  1.841939  ]],
        dtype=tf.float32
    )

    seed = 42
    kwds = dict(rtol=1e-05, atol=1e-08)

    tf.random.set_seed(seed)
    y_ = dpl.GaussianDropout(rate=0.2, is_monte_carlo=False, scale_during_training=False, seed=seed)(a)
    assert_allclose(y_, reference_rate20_nomc_noscale, **kwds)

    tf.random.set_seed(seed)
    y_ = dpl.GaussianDropout(rate=0.2, is_monte_carlo=True, scale_during_training=False, seed=seed)(a)
    assert_allclose(y_, reference_rate20_mc_noscale, **kwds)

    tf.random.set_seed(seed)
    y_ = dpl.GaussianDropout(rate=0.2, is_monte_carlo=False, scale_during_training=True, seed=seed)(a)
    assert_allclose(y_, reference_rate20_nomc_scale, **kwds)

    tf.random.set_seed(seed)
    y_ = dpl.GaussianDropout(rate=0.2, is_monte_carlo=True, scale_during_training=True, seed=seed)(a)
    assert_allclose(y_, reference_rate20_mc_scale, **kwds)
