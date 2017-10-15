from code_base.classifiers.cnn import ThreeLayerConvNet

Models = {}
Solvers = {}
Models['conv64_filter5_fc512_drop0'] = ThreeLayerConvNet(num_classes=2, weight_scale=0.001, hidden_dim=512, reg=0.0001,
                                                         num_filters=64, filter_size=5, dropout=0)

Models['conv64_filter5_fc512_drop03'] = ThreeLayerConvNet(num_classes=2, weight_scale=0.001, hidden_dim=512, reg=0.0001,
                                                          num_filters=64, filter_size=5, dropout=0.3)

Models['conv128_filter3_fc1024_drop0'] = ThreeLayerConvNet(num_classes=2, weight_scale=0.001, hidden_dim=1024,
                                                           reg=0.0001, num_filters=128, filter_size=3, dropout=0)

Models['conv128_filter3_fc1024_drop04'] = ThreeLayerConvNet(num_classes=2, weight_scale=0.001, hidden_dim=1024,
                                                            reg=0.0001, num_filters=128, filter_size=3, dropout=0.4)

Models['conv32_filter7_fc256_drop0'] = ThreeLayerConvNet(num_classes=2, weight_scale=0.001, hidden_dim=256, reg=0.0001,
                                                         num_filters=32, filter_size=7, dropout=0)

Models['conv32_filter7_fc256_drop02'] = ThreeLayerConvNet(num_classes=2, weight_scale=0.001, hidden_dim=256, reg=0.0001,
                                                          num_filters=32, filter_size=7, dropout=0.2)
