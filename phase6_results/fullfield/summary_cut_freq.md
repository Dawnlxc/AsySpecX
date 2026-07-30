# Phase 4-Finalize Cut_freq Diagnostics

Best cut_freq **by validation** is the selection-safe choice. Best-by-test is shown for analysis only and must NOT drive selection.

## Per cut_freq

| dataset | pred_len | arm | cut_freq | n | val_mse | mse | mae |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| ETTh1 | 192 | phase6_asx_cross | 181 | 3 | 0.960089 | 0.418848 | 0.427555 |
| ETTh1 | 192 | phase6_asx_cross | 25 | 3 | 1.00961 | 0.439685 | 0.427828 |
| ETTh1 | 192 | phase6_asx_cross_clip05 | 181 | 3 | 0.962032 | 0.414458 | 0.423751 |
| ETTh1 | 192 | phase6_asx_cross_clip05 | 25 | 3 | 1.0111 | 0.435523 | 0.425184 |
| ETTh1 | 192 | phase6_asx_individual | 181 | 3 | 1.05167 | 0.432499 | 0.437922 |
| ETTh1 | 192 | phase6_asx_individual | 25 | 3 | 1.01196 | 0.439533 | 0.428551 |
| ETTh1 | 192 | phase6_asx_individual_period | 181 | 3 | 1.04884 | 0.430397 | 0.437061 |
| ETTh1 | 192 | phase6_asx_individual_period | 25 | 3 | 1.01066 | 0.438807 | 0.427583 |
| ETTh1 | 192 | phase6_asx_individual_revin | 181 | 3 | 1.04875 | 0.436349 | 0.442286 |
| ETTh1 | 192 | phase6_asx_individual_revin | 25 | 3 | 1.01794 | 0.438955 | 0.427439 |
| ETTh1 | 192 | phase6_asx_period_multi | 181 | 3 | 0.956738 | 0.416658 | 0.425533 |
| ETTh1 | 192 | phase6_asx_period_multi | 25 | 3 | 1.00668 | 0.437965 | 0.425315 |
| ETTh1 | 336 | phase6_asx_cross | 181 | 3 | 1.15677 | 0.44444 | 0.443511 |
| ETTh1 | 336 | phase6_asx_cross | 25 | 3 | 1.29046 | 0.480089 | 0.445358 |
| ETTh1 | 336 | phase6_asx_cross_clip05 | 181 | 3 | 1.18394 | 0.433757 | 0.438276 |
| ETTh1 | 336 | phase6_asx_cross_clip05 | 25 | 3 | 1.29535 | 0.474925 | 0.445853 |
| ETTh1 | 336 | phase6_asx_individual | 181 | 3 | 1.27142 | 0.454942 | 0.458269 |
| ETTh1 | 336 | phase6_asx_individual | 25 | 3 | 1.31036 | 0.480212 | 0.451452 |
| ETTh1 | 336 | phase6_asx_individual_period | 181 | 3 | 1.26512 | 0.450916 | 0.455386 |
| ETTh1 | 336 | phase6_asx_individual_period | 25 | 3 | 1.30819 | 0.479148 | 0.450119 |
| ETTh1 | 336 | phase6_asx_individual_revin | 181 | 3 | 1.2606 | 0.466571 | 0.470113 |
| ETTh1 | 336 | phase6_asx_individual_revin | 25 | 3 | 1.3216 | 0.479844 | 0.450546 |
| ETTh1 | 336 | phase6_asx_period_multi | 181 | 3 | 1.15617 | 0.442249 | 0.440963 |
| ETTh1 | 336 | phase6_asx_period_multi | 25 | 3 | 1.28815 | 0.478458 | 0.443393 |
| ETTh1 | 720 | phase6_asx_cross | 181 | 3 | 1.39722 | 0.454626 | 0.464548 |
| ETTh1 | 720 | phase6_asx_cross | 25 | 3 | 1.56647 | 0.468941 | 0.457149 |
| ETTh1 | 720 | phase6_asx_cross_clip05 | 181 | 3 | 1.43335 | 0.431043 | 0.455236 |
| ETTh1 | 720 | phase6_asx_cross_clip05 | 25 | 3 | 1.56968 | 0.462566 | 0.462729 |
| ETTh1 | 720 | phase6_asx_individual | 181 | 3 | 1.51996 | 0.456004 | 0.473286 |
| ETTh1 | 720 | phase6_asx_individual | 25 | 3 | 1.58861 | 0.468068 | 0.470318 |
| ETTh1 | 720 | phase6_asx_individual_period | 181 | 3 | 1.51215 | 0.45213 | 0.469842 |
| ETTh1 | 720 | phase6_asx_individual_period | 25 | 3 | 1.58727 | 0.466929 | 0.469099 |
| ETTh1 | 720 | phase6_asx_individual_revin | 181 | 3 | 1.47771 | 0.521563 | 0.516194 |
| ETTh1 | 720 | phase6_asx_individual_revin | 25 | 3 | 1.58575 | 0.481158 | 0.479132 |
| ETTh1 | 720 | phase6_asx_period_multi | 181 | 3 | 1.39234 | 0.448751 | 0.460177 |
| ETTh1 | 720 | phase6_asx_period_multi | 25 | 3 | 1.5633 | 0.467565 | 0.454944 |
| ETTh1 | 96 | phase6_asx_cross | 181 | 3 | 0.713325 | 0.382848 | 0.40744 |
| ETTh1 | 96 | phase6_asx_cross | 25 | 3 | 0.709157 | 0.391569 | 0.400694 |
| ETTh1 | 96 | phase6_asx_cross_clip05 | 181 | 3 | 0.700948 | 0.379997 | 0.403904 |
| ETTh1 | 96 | phase6_asx_cross_clip05 | 25 | 3 | 0.711368 | 0.388208 | 0.397494 |
| ETTh1 | 96 | phase6_asx_individual | 181 | 3 | 0.775744 | 0.399946 | 0.4162 |
| ETTh1 | 96 | phase6_asx_individual | 25 | 3 | 0.716344 | 0.390342 | 0.399283 |
| ETTh1 | 96 | phase6_asx_individual_period | 181 | 3 | 0.774367 | 0.39923 | 0.415608 |
| ETTh1 | 96 | phase6_asx_individual_period | 25 | 3 | 0.715429 | 0.389594 | 0.398505 |
| ETTh1 | 96 | phase6_asx_individual_revin | 181 | 3 | 0.775874 | 0.401902 | 0.418328 |
| ETTh1 | 96 | phase6_asx_individual_revin | 25 | 3 | 0.718565 | 0.390026 | 0.39847 |
| ETTh1 | 96 | phase6_asx_period_multi | 181 | 3 | 0.711173 | 0.381765 | 0.406364 |
| ETTh1 | 96 | phase6_asx_period_multi | 25 | 3 | 0.707168 | 0.389832 | 0.398654 |
| ETTm1 | 192 | phase6_asx_cross | 43 | 3 | 0.51606 | 0.349505 | 0.380245 |
| ETTm1 | 192 | phase6_asx_cross | 7 | 3 | 0.532813 | 0.390432 | 0.397186 |
| ETTm1 | 192 | phase6_asx_cross_clip05 | 43 | 3 | 0.524013 | 0.353204 | 0.382863 |
| ETTm1 | 192 | phase6_asx_cross_clip05 | 7 | 3 | 0.532939 | 0.388281 | 0.394429 |
| ETTm1 | 192 | phase6_asx_individual | 43 | 3 | 0.505655 | 0.345064 | 0.371671 |
| ETTm1 | 192 | phase6_asx_individual | 7 | 3 | 0.531532 | 0.390304 | 0.393907 |
| ETTm1 | 192 | phase6_asx_individual_period | 43 | 3 | 0.504519 | 0.343985 | 0.370746 |
| ETTm1 | 192 | phase6_asx_individual_period | 7 | 3 | 0.53115 | 0.389886 | 0.393571 |
| ETTm1 | 192 | phase6_asx_individual_revin | 43 | 3 | 0.507097 | 0.344973 | 0.371357 |
| ETTm1 | 192 | phase6_asx_individual_revin | 7 | 3 | 0.531832 | 0.389531 | 0.392282 |
| ETTm1 | 192 | phase6_asx_period_multi | 43 | 3 | 0.514229 | 0.346639 | 0.377633 |
| ETTm1 | 192 | phase6_asx_period_multi | 7 | 3 | 0.53364 | 0.390427 | 0.396708 |
| ETTm1 | 336 | phase6_asx_cross | 43 | 3 | 0.661885 | 0.379319 | 0.398616 |
| ETTm1 | 336 | phase6_asx_cross | 7 | 3 | 0.677555 | 0.418064 | 0.414438 |
| ETTm1 | 336 | phase6_asx_cross_clip05 | 43 | 3 | 0.665603 | 0.378102 | 0.398451 |
| ETTm1 | 336 | phase6_asx_cross_clip05 | 7 | 3 | 0.679315 | 0.416828 | 0.412478 |
| ETTm1 | 336 | phase6_asx_individual | 43 | 3 | 0.645113 | 0.376329 | 0.389791 |
| ETTm1 | 336 | phase6_asx_individual | 7 | 3 | 0.67977 | 0.419986 | 0.412859 |
| ETTm1 | 336 | phase6_asx_individual_period | 43 | 3 | 0.643925 | 0.375191 | 0.388813 |
| ETTm1 | 336 | phase6_asx_individual_period | 7 | 3 | 0.679453 | 0.419624 | 0.412574 |
| ETTm1 | 336 | phase6_asx_individual_revin | 43 | 3 | 0.648475 | 0.376144 | 0.389275 |
| ETTm1 | 336 | phase6_asx_individual_revin | 7 | 3 | 0.679868 | 0.418762 | 0.41056 |
| ETTm1 | 336 | phase6_asx_period_multi | 43 | 3 | 0.659497 | 0.37667 | 0.395604 |
| ETTm1 | 336 | phase6_asx_period_multi | 7 | 3 | 0.678049 | 0.417645 | 0.413793 |
| ETTm1 | 720 | phase6_asx_cross | 43 | 3 | 0.936525 | 0.424793 | 0.422255 |
| ETTm1 | 720 | phase6_asx_cross | 7 | 3 | 0.994029 | 0.482212 | 0.447196 |
| ETTm1 | 720 | phase6_asx_cross_clip05 | 43 | 3 | 0.937028 | 0.42411 | 0.42207 |
| ETTm1 | 720 | phase6_asx_cross_clip05 | 7 | 3 | 0.995679 | 0.478628 | 0.444328 |
| ETTm1 | 720 | phase6_asx_individual | 43 | 3 | 0.931924 | 0.426381 | 0.417995 |
| ETTm1 | 720 | phase6_asx_individual | 7 | 3 | 0.996871 | 0.482827 | 0.445749 |
| ETTm1 | 720 | phase6_asx_individual_period | 43 | 3 | 0.930672 | 0.425106 | 0.416921 |
| ETTm1 | 720 | phase6_asx_individual_period | 7 | 3 | 0.996606 | 0.482527 | 0.445505 |
| ETTm1 | 720 | phase6_asx_individual_revin | 43 | 3 | 0.938547 | 0.426514 | 0.417585 |
| ETTm1 | 720 | phase6_asx_individual_revin | 7 | 3 | 0.998319 | 0.480535 | 0.44272 |
| ETTm1 | 720 | phase6_asx_period_multi | 43 | 3 | 0.936706 | 0.423044 | 0.419741 |
| ETTm1 | 720 | phase6_asx_period_multi | 7 | 3 | 0.993906 | 0.481556 | 0.446495 |
| ETTm1 | 96 | phase6_asx_cross | 43 | 3 | 0.414106 | 0.318785 | 0.361816 |
| ETTm1 | 96 | phase6_asx_cross | 7 | 3 | 0.411029 | 0.356121 | 0.382839 |
| ETTm1 | 96 | phase6_asx_cross_clip05 | 43 | 3 | 0.417647 | 0.31996 | 0.363671 |
| ETTm1 | 96 | phase6_asx_cross_clip05 | 7 | 3 | 0.405896 | 0.351353 | 0.379215 |
| ETTm1 | 96 | phase6_asx_individual | 43 | 3 | 0.391794 | 0.30783 | 0.352164 |
| ETTm1 | 96 | phase6_asx_individual | 7 | 3 | 0.407939 | 0.356278 | 0.380528 |
| ETTm1 | 96 | phase6_asx_individual_period | 43 | 3 | 0.390972 | 0.307015 | 0.351471 |
| ETTm1 | 96 | phase6_asx_individual_period | 7 | 3 | 0.407419 | 0.355771 | 0.380156 |
| ETTm1 | 96 | phase6_asx_individual_revin | 43 | 3 | 0.392346 | 0.307806 | 0.352017 |
| ETTm1 | 96 | phase6_asx_individual_revin | 7 | 3 | 0.408788 | 0.356269 | 0.379725 |
| ETTm1 | 96 | phase6_asx_period_multi | 43 | 3 | 0.411422 | 0.316857 | 0.360388 |
| ETTm1 | 96 | phase6_asx_period_multi | 7 | 3 | 0.409117 | 0.354767 | 0.381628 |
| PEMS04 | 12 | phase6_asx_cross | 49 | 3 | 0.0976024 | 0.093211 | 0.204826 |
| PEMS04 | 12 | phase6_asx_cross_clip05 | 49 | 3 | 0.100825 | 0.0956966 | 0.206388 |
| PEMS04 | 12 | phase6_asx_individual | 49 | 3 | 0.12897 | 0.121114 | 0.232542 |
| PEMS04 | 12 | phase6_asx_individual_period | 49 | 3 | 0.128513 | 0.120684 | 0.231889 |
| PEMS04 | 12 | phase6_asx_individual_revin | 49 | 3 | 0.12897 | 0.121113 | 0.232542 |
| PEMS04 | 12 | phase6_asx_period_multi | 49 | 3 | 0.0974851 | 0.0931334 | 0.204459 |
| PEMS04 | 24 | phase6_asx_cross | 49 | 3 | 0.144523 | 0.139606 | 0.256724 |
| PEMS04 | 24 | phase6_asx_cross_clip05 | 49 | 3 | 0.158534 | 0.15063 | 0.26438 |
| PEMS04 | 24 | phase6_asx_individual | 49 | 3 | 0.245408 | 0.225591 | 0.322719 |
| PEMS04 | 24 | phase6_asx_individual_period | 49 | 3 | 0.244044 | 0.224256 | 0.321125 |
| PEMS04 | 24 | phase6_asx_individual_revin | 49 | 3 | 0.245405 | 0.225588 | 0.322722 |
| PEMS04 | 24 | phase6_asx_period_multi | 49 | 3 | 0.143597 | 0.138738 | 0.255873 |
| PEMS04 | 48 | phase6_asx_cross | 49 | 3 | 0.261134 | 0.251873 | 0.357518 |
| PEMS04 | 48 | phase6_asx_cross_clip05 | 49 | 3 | 0.302828 | 0.290361 | 0.381654 |
| PEMS04 | 48 | phase6_asx_individual | 49 | 3 | 0.553549 | 0.505913 | 0.505226 |
| PEMS04 | 48 | phase6_asx_individual_period | 49 | 3 | 0.550471 | 0.502697 | 0.502698 |
| PEMS04 | 48 | phase6_asx_individual_revin | 49 | 3 | 0.553477 | 0.505883 | 0.505208 |
| PEMS04 | 48 | phase6_asx_period_multi | 49 | 3 | 0.258745 | 0.249868 | 0.355835 |
| PEMS04 | 96 | phase6_asx_cross | 49 | 3 | 0.424246 | 0.393524 | 0.464234 |
| PEMS04 | 96 | phase6_asx_cross_clip05 | 49 | 3 | 0.651982 | 0.612718 | 0.587717 |
| PEMS04 | 96 | phase6_asx_individual | 49 | 3 | 1.11841 | 1.01435 | 0.762649 |
| PEMS04 | 96 | phase6_asx_individual_period | 49 | 3 | 1.11433 | 1.00813 | 0.758682 |
| PEMS04 | 96 | phase6_asx_individual_revin | 49 | 3 | 1.11445 | 1.01151 | 0.760199 |
| PEMS04 | 96 | phase6_asx_period_multi | 49 | 3 | 0.405152 | 0.381494 | 0.455779 |
| PEMS08 | 12 | phase6_asx_cross | 49 | 3 | 0.101567 | 0.0921295 | 0.200012 |
| PEMS08 | 12 | phase6_asx_cross_clip05 | 49 | 3 | 0.103603 | 0.0947272 | 0.203666 |
| PEMS08 | 12 | phase6_asx_individual | 49 | 3 | 0.131218 | 0.117777 | 0.227301 |
| PEMS08 | 12 | phase6_asx_individual_period | 49 | 3 | 0.130818 | 0.117358 | 0.226733 |
| PEMS08 | 12 | phase6_asx_individual_revin | 49 | 3 | 0.131218 | 0.117777 | 0.227301 |
| PEMS08 | 12 | phase6_asx_period_multi | 49 | 3 | 0.100835 | 0.0914837 | 0.199324 |
| PEMS08 | 24 | phase6_asx_cross | 49 | 3 | 0.16224 | 0.150745 | 0.258782 |
| PEMS08 | 24 | phase6_asx_cross_clip05 | 49 | 3 | 0.169788 | 0.158126 | 0.267629 |
| PEMS08 | 24 | phase6_asx_individual | 49 | 3 | 0.248014 | 0.223927 | 0.319327 |
| PEMS08 | 24 | phase6_asx_individual_period | 49 | 3 | 0.24676 | 0.222632 | 0.317932 |
| PEMS08 | 24 | phase6_asx_individual_revin | 49 | 3 | 0.248015 | 0.223925 | 0.319327 |
| PEMS08 | 24 | phase6_asx_period_multi | 49 | 3 | 0.160876 | 0.149582 | 0.257745 |
| PEMS08 | 48 | phase6_asx_cross | 49 | 3 | 0.294638 | 0.301337 | 0.386802 |
| PEMS08 | 48 | phase6_asx_cross_clip05 | 49 | 3 | 0.335884 | 0.324344 | 0.402917 |
| PEMS08 | 48 | phase6_asx_individual | 49 | 3 | 0.564816 | 0.522667 | 0.512322 |
| PEMS08 | 48 | phase6_asx_individual_period | 49 | 3 | 0.561925 | 0.519624 | 0.510045 |
| PEMS08 | 48 | phase6_asx_individual_revin | 49 | 3 | 0.564654 | 0.522536 | 0.512015 |
| PEMS08 | 48 | phase6_asx_period_multi | 49 | 3 | 0.286323 | 0.29149 | 0.37999 |
| PEMS08 | 96 | phase6_asx_cross | 49 | 3 | 0.523604 | 0.579403 | 0.555761 |
| PEMS08 | 96 | phase6_asx_cross_clip05 | 49 | 3 | 0.718902 | 0.717465 | 0.628071 |
| PEMS08 | 96 | phase6_asx_individual | 49 | 3 | 1.16385 | 1.11728 | 0.786571 |
| PEMS08 | 96 | phase6_asx_individual_period | 49 | 3 | 1.15986 | 1.11244 | 0.782991 |
| PEMS08 | 96 | phase6_asx_individual_revin | 49 | 3 | 1.15703 | 1.11243 | 0.780548 |
| PEMS08 | 96 | phase6_asx_period_multi | 49 | 3 | 0.541883 | 0.561115 | 0.545434 |
| electricity | 192 | phase6_asx_cross | 181 | 3 | 0.131551 | 0.156384 | 0.25649 |
| electricity | 192 | phase6_asx_cross | 25 | 3 | 0.177014 | 0.202865 | 0.290725 |
| electricity | 192 | phase6_asx_cross_clip05 | 181 | 3 | 0.130993 | 0.155345 | 0.256769 |
| electricity | 192 | phase6_asx_cross_clip05 | 25 | 3 | 0.174956 | 0.200751 | 0.288115 |
| electricity | 192 | phase6_asx_individual | 181 | 3 | 0.132576 | 0.15903 | 0.258117 |
| electricity | 192 | phase6_asx_individual | 25 | 3 | 0.170741 | 0.195592 | 0.28256 |
| electricity | 192 | phase6_asx_individual_period | 181 | 3 | 0.131225 | 0.157527 | 0.255902 |
| electricity | 192 | phase6_asx_individual_period | 25 | 3 | 0.169799 | 0.194564 | 0.281049 |
| electricity | 192 | phase6_asx_individual_revin | 181 | 3 | 0.132509 | 0.15915 | 0.257916 |
| electricity | 192 | phase6_asx_individual_revin | 25 | 3 | 0.17024 | 0.195396 | 0.281997 |
| electricity | 192 | phase6_asx_period_multi | 181 | 3 | 0.127752 | 0.151979 | 0.249248 |
| electricity | 192 | phase6_asx_period_multi | 25 | 3 | 0.17349 | 0.19882 | 0.283511 |
| electricity | 336 | phase6_asx_cross | 181 | 3 | 0.146601 | 0.171584 | 0.271645 |
| electricity | 336 | phase6_asx_cross | 25 | 3 | 0.191162 | 0.217172 | 0.305468 |
| electricity | 336 | phase6_asx_cross_clip05 | 181 | 3 | 0.147717 | 0.171554 | 0.272854 |
| electricity | 336 | phase6_asx_cross_clip05 | 25 | 3 | 0.189343 | 0.215134 | 0.302338 |
| electricity | 336 | phase6_asx_individual | 181 | 3 | 0.147573 | 0.173817 | 0.27306 |
| electricity | 336 | phase6_asx_individual | 25 | 3 | 0.184864 | 0.210002 | 0.296637 |
| electricity | 336 | phase6_asx_individual_period | 181 | 3 | 0.145905 | 0.171962 | 0.27044 |
| electricity | 336 | phase6_asx_individual_period | 25 | 3 | 0.183885 | 0.208904 | 0.295086 |
| electricity | 336 | phase6_asx_individual_revin | 181 | 3 | 0.147353 | 0.174186 | 0.27284 |
| electricity | 336 | phase6_asx_individual_revin | 25 | 3 | 0.184209 | 0.209877 | 0.29599 |
| electricity | 336 | phase6_asx_period_multi | 181 | 3 | 0.142607 | 0.166828 | 0.263942 |
| electricity | 336 | phase6_asx_period_multi | 25 | 3 | 0.187399 | 0.212771 | 0.298486 |
| electricity | 720 | phase6_asx_cross | 181 | 3 | 0.178851 | 0.209717 | 0.304132 |
| electricity | 720 | phase6_asx_cross | 25 | 3 | 0.223918 | 0.256258 | 0.335199 |
| electricity | 720 | phase6_asx_cross_clip05 | 181 | 3 | 0.181238 | 0.206815 | 0.30179 |
| electricity | 720 | phase6_asx_cross_clip05 | 25 | 3 | 0.221622 | 0.253832 | 0.332686 |
| electricity | 720 | phase6_asx_individual | 181 | 3 | 0.179707 | 0.209144 | 0.302893 |
| electricity | 720 | phase6_asx_individual | 25 | 3 | 0.218405 | 0.250897 | 0.328754 |
| electricity | 720 | phase6_asx_individual_period | 181 | 3 | 0.177137 | 0.206207 | 0.2989 |
| electricity | 720 | phase6_asx_individual_period | 25 | 3 | 0.217526 | 0.249786 | 0.327229 |
| electricity | 720 | phase6_asx_individual_revin | 181 | 3 | 0.17811 | 0.209677 | 0.302399 |
| electricity | 720 | phase6_asx_individual_revin | 25 | 3 | 0.216667 | 0.25078 | 0.327686 |
| electricity | 720 | phase6_asx_period_multi | 181 | 3 | 0.174302 | 0.203556 | 0.293765 |
| electricity | 720 | phase6_asx_period_multi | 25 | 3 | 0.22126 | 0.253064 | 0.330022 |
| electricity | 96 | phase6_asx_cross | 181 | 3 | 0.119219 | 0.141479 | 0.243351 |
| electricity | 96 | phase6_asx_cross | 25 | 3 | 0.179661 | 0.203362 | 0.289226 |
| electricity | 96 | phase6_asx_cross_clip05 | 181 | 3 | 0.119289 | 0.142084 | 0.244734 |
| electricity | 96 | phase6_asx_cross_clip05 | 25 | 3 | 0.176535 | 0.200152 | 0.286413 |
| electricity | 96 | phase6_asx_individual | 181 | 3 | 0.11977 | 0.142635 | 0.244435 |
| electricity | 96 | phase6_asx_individual | 25 | 3 | 0.172418 | 0.195684 | 0.281415 |
| electricity | 96 | phase6_asx_individual_period | 181 | 3 | 0.11868 | 0.141463 | 0.24263 |
| electricity | 96 | phase6_asx_individual_period | 25 | 3 | 0.1715 | 0.194684 | 0.279893 |
| electricity | 96 | phase6_asx_individual_revin | 181 | 3 | 0.119742 | 0.142674 | 0.244322 |
| electricity | 96 | phase6_asx_individual_revin | 25 | 3 | 0.17201 | 0.195511 | 0.280829 |
| electricity | 96 | phase6_asx_period_multi | 181 | 3 | 0.116401 | 0.138265 | 0.238301 |
| electricity | 96 | phase6_asx_period_multi | 25 | 3 | 0.176796 | 0.200152 | 0.283614 |
| traffic | 192 | phase6_asx_cross | 181 | 3 | 0.335048 | 0.405761 | 0.285326 |
| traffic | 192 | phase6_asx_cross | 25 | 3 | 0.497728 | 0.619871 | 0.383648 |
| traffic | 192 | phase6_asx_cross_clip05 | 181 | 3 | 0.334217 | 0.403841 | 0.286869 |
| traffic | 192 | phase6_asx_cross_clip05 | 25 | 3 | 0.47703 | 0.584051 | 0.370466 |
| traffic | 192 | phase6_asx_individual | 181 | 3 | 0.347328 | 0.425683 | 0.303913 |
| traffic | 192 | phase6_asx_individual | 25 | 3 | 0.492735 | 0.60884 | 0.378827 |
| traffic | 192 | phase6_asx_individual_period | 181 | 3 | 0.346323 | 0.424449 | 0.302633 |
| traffic | 192 | phase6_asx_individual_period | 25 | 3 | 0.4917 | 0.607659 | 0.37709 |
| traffic | 192 | phase6_asx_individual_revin | 181 | 3 | 0.347856 | 0.425921 | 0.304208 |
| traffic | 192 | phase6_asx_individual_revin | 25 | 3 | 0.492337 | 0.608354 | 0.377015 |
| traffic | 192 | phase6_asx_period_multi | 181 | 3 | 0.332126 | 0.400557 | 0.281291 |
| traffic | 192 | phase6_asx_period_multi | 25 | 3 | 0.491185 | 0.607545 | 0.377175 |
| traffic | 336 | phase6_asx_cross | 181 | 3 | 0.34709 | 0.41926 | 0.291002 |
| traffic | 336 | phase6_asx_cross | 25 | 3 | 0.492137 | 0.634431 | 0.385272 |
| traffic | 336 | phase6_asx_cross_clip05 | 181 | 3 | 0.347172 | 0.417442 | 0.293291 |
| traffic | 336 | phase6_asx_cross_clip05 | 25 | 3 | 0.478681 | 0.590953 | 0.37248 |
| traffic | 336 | phase6_asx_individual | 181 | 3 | 0.358041 | 0.438215 | 0.308664 |
| traffic | 336 | phase6_asx_individual | 25 | 3 | 0.49272 | 0.615203 | 0.381239 |
| traffic | 336 | phase6_asx_individual_period | 181 | 3 | 0.35677 | 0.43679 | 0.307201 |
| traffic | 336 | phase6_asx_individual_period | 25 | 3 | 0.491505 | 0.613966 | 0.379375 |
| traffic | 336 | phase6_asx_individual_revin | 181 | 3 | 0.359312 | 0.438476 | 0.309082 |
| traffic | 336 | phase6_asx_individual_revin | 25 | 3 | 0.492347 | 0.614833 | 0.37944 |
| traffic | 336 | phase6_asx_period_multi | 181 | 3 | 0.343343 | 0.411927 | 0.285796 |
| traffic | 336 | phase6_asx_period_multi | 25 | 3 | 0.486068 | 0.619659 | 0.378601 |
| traffic | 720 | phase6_asx_cross | 181 | 3 | 0.395136 | 0.472758 | 0.317126 |
| traffic | 720 | phase6_asx_cross | 25 | 3 | 0.540626 | 0.668825 | 0.401002 |
| traffic | 720 | phase6_asx_cross_clip05 | 181 | 3 | 0.395733 | 0.457618 | 0.318811 |
| traffic | 720 | phase6_asx_cross_clip05 | 25 | 3 | 0.529383 | 0.624815 | 0.388458 |
| traffic | 720 | phase6_asx_individual | 181 | 3 | 0.406701 | 0.476338 | 0.327656 |
| traffic | 720 | phase6_asx_individual | 25 | 3 | 0.545915 | 0.654848 | 0.403148 |
| traffic | 720 | phase6_asx_individual_period | 181 | 3 | 0.404594 | 0.474066 | 0.325096 |
| traffic | 720 | phase6_asx_individual_period | 25 | 3 | 0.54428 | 0.653022 | 0.400383 |
| traffic | 720 | phase6_asx_individual_revin | 181 | 3 | 0.408481 | 0.476954 | 0.328577 |
| traffic | 720 | phase6_asx_individual_revin | 25 | 3 | 0.545667 | 0.654684 | 0.400752 |
| traffic | 720 | phase6_asx_period_multi | 181 | 3 | 0.386709 | 0.448077 | 0.304166 |
| traffic | 720 | phase6_asx_period_multi | 25 | 3 | 0.53645 | 0.656346 | 0.395585 |
| traffic | 96 | phase6_asx_cross | 181 | 3 | 0.329335 | 0.390343 | 0.279852 |
| traffic | 96 | phase6_asx_cross | 25 | 3 | 0.545891 | 0.668749 | 0.405404 |
| traffic | 96 | phase6_asx_cross_clip05 | 181 | 3 | 0.329242 | 0.390999 | 0.280972 |
| traffic | 96 | phase6_asx_cross_clip05 | 25 | 3 | 0.519353 | 0.621603 | 0.392252 |
| traffic | 96 | phase6_asx_individual | 181 | 3 | 0.346612 | 0.416057 | 0.301387 |
| traffic | 96 | phase6_asx_individual | 25 | 3 | 0.545929 | 0.658153 | 0.402363 |
| traffic | 96 | phase6_asx_individual_period | 181 | 3 | 0.34586 | 0.415094 | 0.30037 |
| traffic | 96 | phase6_asx_individual_period | 25 | 3 | 0.545228 | 0.657261 | 0.401295 |
| traffic | 96 | phase6_asx_individual_revin | 181 | 3 | 0.346855 | 0.416248 | 0.301584 |
| traffic | 96 | phase6_asx_individual_revin | 25 | 3 | 0.545304 | 0.657528 | 0.400171 |
| traffic | 96 | phase6_asx_period_multi | 181 | 3 | 0.328253 | 0.388673 | 0.278225 |
| traffic | 96 | phase6_asx_period_multi | 25 | 3 | 0.542161 | 0.660897 | 0.401897 |
| weather | 192 | phase6_asx_cross | 2 | 3 | 0.540701 | 0.243288 | 0.285846 |
| weather | 192 | phase6_asx_cross | 31 | 3 | 0.437878 | 0.197634 | 0.250958 |
| weather | 192 | phase6_asx_cross_clip05 | 2 | 3 | 0.546581 | 0.246579 | 0.286694 |
| weather | 192 | phase6_asx_cross_clip05 | 31 | 3 | 0.443234 | 0.203098 | 0.256164 |
| weather | 192 | phase6_asx_individual | 2 | 3 | 0.54041 | 0.23555 | 0.279369 |
| weather | 192 | phase6_asx_individual | 31 | 3 | 0.438676 | 0.190698 | 0.242888 |
| weather | 192 | phase6_asx_individual_period | 2 | 3 | 0.54037 | 0.235442 | 0.27906 |
| weather | 192 | phase6_asx_individual_period | 31 | 3 | 0.438569 | 0.190687 | 0.242814 |
| weather | 192 | phase6_asx_individual_revin | 2 | 3 | 0.52831 | 0.231881 | 0.274434 |
| weather | 192 | phase6_asx_individual_revin | 31 | 3 | 0.437773 | 0.190203 | 0.240335 |
| weather | 192 | phase6_asx_period_multi | 2 | 3 | 0.538942 | 0.242643 | 0.285154 |
| weather | 192 | phase6_asx_period_multi | 31 | 3 | 0.438096 | 0.197595 | 0.250832 |
| weather | 336 | phase6_asx_cross | 2 | 3 | 0.627257 | 0.294065 | 0.318372 |
| weather | 336 | phase6_asx_cross | 31 | 3 | 0.499075 | 0.249578 | 0.290083 |
| weather | 336 | phase6_asx_cross_clip05 | 2 | 3 | 0.631273 | 0.296217 | 0.318544 |
| weather | 336 | phase6_asx_cross_clip05 | 31 | 3 | 0.504605 | 0.252263 | 0.291876 |
| weather | 336 | phase6_asx_individual | 2 | 3 | 0.623713 | 0.287709 | 0.314102 |
| weather | 336 | phase6_asx_individual | 31 | 3 | 0.503298 | 0.240569 | 0.282137 |
| weather | 336 | phase6_asx_individual_period | 2 | 3 | 0.623399 | 0.28755 | 0.313749 |
| weather | 336 | phase6_asx_individual_period | 31 | 3 | 0.503263 | 0.240582 | 0.282093 |
| weather | 336 | phase6_asx_individual_revin | 2 | 3 | 0.605493 | 0.283318 | 0.308607 |
| weather | 336 | phase6_asx_individual_revin | 31 | 3 | 0.502111 | 0.240016 | 0.279124 |
| weather | 336 | phase6_asx_period_multi | 2 | 3 | 0.626016 | 0.293859 | 0.318049 |
| weather | 336 | phase6_asx_period_multi | 31 | 3 | 0.499061 | 0.249379 | 0.28983 |
| weather | 720 | phase6_asx_cross | 2 | 3 | 0.753096 | 0.36506 | 0.361983 |
| weather | 720 | phase6_asx_cross | 31 | 3 | 0.591443 | 0.312987 | 0.335959 |
| weather | 720 | phase6_asx_cross_clip05 | 2 | 3 | 0.758068 | 0.36717 | 0.362145 |
| weather | 720 | phase6_asx_cross_clip05 | 31 | 3 | 0.59686 | 0.318011 | 0.338353 |
| weather | 720 | phase6_asx_individual | 2 | 3 | 0.745751 | 0.361929 | 0.359699 |
| weather | 720 | phase6_asx_individual | 31 | 3 | 0.595775 | 0.309688 | 0.331898 |
| weather | 720 | phase6_asx_individual_period | 2 | 3 | 0.745231 | 0.361695 | 0.35926 |
| weather | 720 | phase6_asx_individual_period | 31 | 3 | 0.595722 | 0.309708 | 0.331879 |
| weather | 720 | phase6_asx_individual_revin | 2 | 3 | 0.724068 | 0.3567 | 0.353665 |
| weather | 720 | phase6_asx_individual_revin | 31 | 3 | 0.596845 | 0.31011 | 0.328973 |
| weather | 720 | phase6_asx_period_multi | 2 | 3 | 0.752222 | 0.364729 | 0.361471 |
| weather | 720 | phase6_asx_period_multi | 31 | 3 | 0.591314 | 0.313159 | 0.336044 |
| weather | 96 | phase6_asx_cross | 2 | 3 | 0.467359 | 0.191372 | 0.24636 |
| weather | 96 | phase6_asx_cross | 31 | 3 | 0.387855 | 0.152079 | 0.20958 |
| weather | 96 | phase6_asx_cross_clip05 | 2 | 3 | 0.479362 | 0.200547 | 0.251191 |
| weather | 96 | phase6_asx_cross_clip05 | 31 | 3 | 0.38526 | 0.157925 | 0.217216 |
| weather | 96 | phase6_asx_individual | 2 | 3 | 0.470847 | 0.18987 | 0.24346 |
| weather | 96 | phase6_asx_individual | 31 | 3 | 0.382135 | 0.148135 | 0.202349 |
| weather | 96 | phase6_asx_individual_period | 2 | 3 | 0.470698 | 0.189862 | 0.243297 |
| weather | 96 | phase6_asx_individual_period | 31 | 3 | 0.382017 | 0.148103 | 0.20224 |
| weather | 96 | phase6_asx_individual_revin | 2 | 3 | 0.459084 | 0.187501 | 0.239589 |
| weather | 96 | phase6_asx_individual_revin | 31 | 3 | 0.381475 | 0.147851 | 0.200624 |
| weather | 96 | phase6_asx_period_multi | 2 | 3 | 0.467382 | 0.191152 | 0.245717 |
| weather | 96 | phase6_asx_period_multi | 31 | 3 | 0.387529 | 0.151885 | 0.209185 |

## Best cut_freq by validation (selection-safe)

| dataset | pred_len | arm | best_cut_freq_val | val_mse |
| --- | ---: | --- | ---: | ---: |
| ETTh1 | 192 | phase6_asx_cross | 181 | 0.960089 |
| ETTh1 | 192 | phase6_asx_cross_clip05 | 181 | 0.962032 |
| ETTh1 | 192 | phase6_asx_individual | 25 | 1.01196 |
| ETTh1 | 192 | phase6_asx_individual_period | 25 | 1.01066 |
| ETTh1 | 192 | phase6_asx_individual_revin | 25 | 1.01794 |
| ETTh1 | 192 | phase6_asx_period_multi | 181 | 0.956738 |
| ETTh1 | 336 | phase6_asx_cross | 181 | 1.15677 |
| ETTh1 | 336 | phase6_asx_cross_clip05 | 181 | 1.18394 |
| ETTh1 | 336 | phase6_asx_individual | 181 | 1.27142 |
| ETTh1 | 336 | phase6_asx_individual_period | 181 | 1.26512 |
| ETTh1 | 336 | phase6_asx_individual_revin | 181 | 1.2606 |
| ETTh1 | 336 | phase6_asx_period_multi | 181 | 1.15617 |
| ETTh1 | 720 | phase6_asx_cross | 181 | 1.39722 |
| ETTh1 | 720 | phase6_asx_cross_clip05 | 181 | 1.43335 |
| ETTh1 | 720 | phase6_asx_individual | 181 | 1.51996 |
| ETTh1 | 720 | phase6_asx_individual_period | 181 | 1.51215 |
| ETTh1 | 720 | phase6_asx_individual_revin | 181 | 1.47771 |
| ETTh1 | 720 | phase6_asx_period_multi | 181 | 1.39234 |
| ETTh1 | 96 | phase6_asx_cross | 25 | 0.709157 |
| ETTh1 | 96 | phase6_asx_cross_clip05 | 181 | 0.700948 |
| ETTh1 | 96 | phase6_asx_individual | 25 | 0.716344 |
| ETTh1 | 96 | phase6_asx_individual_period | 25 | 0.715429 |
| ETTh1 | 96 | phase6_asx_individual_revin | 25 | 0.718565 |
| ETTh1 | 96 | phase6_asx_period_multi | 25 | 0.707168 |
| ETTm1 | 192 | phase6_asx_cross | 43 | 0.51606 |
| ETTm1 | 192 | phase6_asx_cross_clip05 | 43 | 0.524013 |
| ETTm1 | 192 | phase6_asx_individual | 43 | 0.505655 |
| ETTm1 | 192 | phase6_asx_individual_period | 43 | 0.504519 |
| ETTm1 | 192 | phase6_asx_individual_revin | 43 | 0.507097 |
| ETTm1 | 192 | phase6_asx_period_multi | 43 | 0.514229 |
| ETTm1 | 336 | phase6_asx_cross | 43 | 0.661885 |
| ETTm1 | 336 | phase6_asx_cross_clip05 | 43 | 0.665603 |
| ETTm1 | 336 | phase6_asx_individual | 43 | 0.645113 |
| ETTm1 | 336 | phase6_asx_individual_period | 43 | 0.643925 |
| ETTm1 | 336 | phase6_asx_individual_revin | 43 | 0.648475 |
| ETTm1 | 336 | phase6_asx_period_multi | 43 | 0.659497 |
| ETTm1 | 720 | phase6_asx_cross | 43 | 0.936525 |
| ETTm1 | 720 | phase6_asx_cross_clip05 | 43 | 0.937028 |
| ETTm1 | 720 | phase6_asx_individual | 43 | 0.931924 |
| ETTm1 | 720 | phase6_asx_individual_period | 43 | 0.930672 |
| ETTm1 | 720 | phase6_asx_individual_revin | 43 | 0.938547 |
| ETTm1 | 720 | phase6_asx_period_multi | 43 | 0.936706 |
| ETTm1 | 96 | phase6_asx_cross | 7 | 0.411029 |
| ETTm1 | 96 | phase6_asx_cross_clip05 | 7 | 0.405896 |
| ETTm1 | 96 | phase6_asx_individual | 43 | 0.391794 |
| ETTm1 | 96 | phase6_asx_individual_period | 43 | 0.390972 |
| ETTm1 | 96 | phase6_asx_individual_revin | 43 | 0.392346 |
| ETTm1 | 96 | phase6_asx_period_multi | 7 | 0.409117 |
| PEMS04 | 12 | phase6_asx_cross | 49 | 0.0976024 |
| PEMS04 | 12 | phase6_asx_cross_clip05 | 49 | 0.100825 |
| PEMS04 | 12 | phase6_asx_individual | 49 | 0.12897 |
| PEMS04 | 12 | phase6_asx_individual_period | 49 | 0.128513 |
| PEMS04 | 12 | phase6_asx_individual_revin | 49 | 0.12897 |
| PEMS04 | 12 | phase6_asx_period_multi | 49 | 0.0974851 |
| PEMS04 | 24 | phase6_asx_cross | 49 | 0.144523 |
| PEMS04 | 24 | phase6_asx_cross_clip05 | 49 | 0.158534 |
| PEMS04 | 24 | phase6_asx_individual | 49 | 0.245408 |
| PEMS04 | 24 | phase6_asx_individual_period | 49 | 0.244044 |
| PEMS04 | 24 | phase6_asx_individual_revin | 49 | 0.245405 |
| PEMS04 | 24 | phase6_asx_period_multi | 49 | 0.143597 |
| PEMS04 | 48 | phase6_asx_cross | 49 | 0.261134 |
| PEMS04 | 48 | phase6_asx_cross_clip05 | 49 | 0.302828 |
| PEMS04 | 48 | phase6_asx_individual | 49 | 0.553549 |
| PEMS04 | 48 | phase6_asx_individual_period | 49 | 0.550471 |
| PEMS04 | 48 | phase6_asx_individual_revin | 49 | 0.553477 |
| PEMS04 | 48 | phase6_asx_period_multi | 49 | 0.258745 |
| PEMS04 | 96 | phase6_asx_cross | 49 | 0.424246 |
| PEMS04 | 96 | phase6_asx_cross_clip05 | 49 | 0.651982 |
| PEMS04 | 96 | phase6_asx_individual | 49 | 1.11841 |
| PEMS04 | 96 | phase6_asx_individual_period | 49 | 1.11433 |
| PEMS04 | 96 | phase6_asx_individual_revin | 49 | 1.11445 |
| PEMS04 | 96 | phase6_asx_period_multi | 49 | 0.405152 |
| PEMS08 | 12 | phase6_asx_cross | 49 | 0.101567 |
| PEMS08 | 12 | phase6_asx_cross_clip05 | 49 | 0.103603 |
| PEMS08 | 12 | phase6_asx_individual | 49 | 0.131218 |
| PEMS08 | 12 | phase6_asx_individual_period | 49 | 0.130818 |
| PEMS08 | 12 | phase6_asx_individual_revin | 49 | 0.131218 |
| PEMS08 | 12 | phase6_asx_period_multi | 49 | 0.100835 |
| PEMS08 | 24 | phase6_asx_cross | 49 | 0.16224 |
| PEMS08 | 24 | phase6_asx_cross_clip05 | 49 | 0.169788 |
| PEMS08 | 24 | phase6_asx_individual | 49 | 0.248014 |
| PEMS08 | 24 | phase6_asx_individual_period | 49 | 0.24676 |
| PEMS08 | 24 | phase6_asx_individual_revin | 49 | 0.248015 |
| PEMS08 | 24 | phase6_asx_period_multi | 49 | 0.160876 |
| PEMS08 | 48 | phase6_asx_cross | 49 | 0.294638 |
| PEMS08 | 48 | phase6_asx_cross_clip05 | 49 | 0.335884 |
| PEMS08 | 48 | phase6_asx_individual | 49 | 0.564816 |
| PEMS08 | 48 | phase6_asx_individual_period | 49 | 0.561925 |
| PEMS08 | 48 | phase6_asx_individual_revin | 49 | 0.564654 |
| PEMS08 | 48 | phase6_asx_period_multi | 49 | 0.286323 |
| PEMS08 | 96 | phase6_asx_cross | 49 | 0.523604 |
| PEMS08 | 96 | phase6_asx_cross_clip05 | 49 | 0.718902 |
| PEMS08 | 96 | phase6_asx_individual | 49 | 1.16385 |
| PEMS08 | 96 | phase6_asx_individual_period | 49 | 1.15986 |
| PEMS08 | 96 | phase6_asx_individual_revin | 49 | 1.15703 |
| PEMS08 | 96 | phase6_asx_period_multi | 49 | 0.541883 |
| electricity | 192 | phase6_asx_cross | 181 | 0.131551 |
| electricity | 192 | phase6_asx_cross_clip05 | 181 | 0.130993 |
| electricity | 192 | phase6_asx_individual | 181 | 0.132576 |
| electricity | 192 | phase6_asx_individual_period | 181 | 0.131225 |
| electricity | 192 | phase6_asx_individual_revin | 181 | 0.132509 |
| electricity | 192 | phase6_asx_period_multi | 181 | 0.127752 |
| electricity | 336 | phase6_asx_cross | 181 | 0.146601 |
| electricity | 336 | phase6_asx_cross_clip05 | 181 | 0.147717 |
| electricity | 336 | phase6_asx_individual | 181 | 0.147573 |
| electricity | 336 | phase6_asx_individual_period | 181 | 0.145905 |
| electricity | 336 | phase6_asx_individual_revin | 181 | 0.147353 |
| electricity | 336 | phase6_asx_period_multi | 181 | 0.142607 |
| electricity | 720 | phase6_asx_cross | 181 | 0.178851 |
| electricity | 720 | phase6_asx_cross_clip05 | 181 | 0.181238 |
| electricity | 720 | phase6_asx_individual | 181 | 0.179707 |
| electricity | 720 | phase6_asx_individual_period | 181 | 0.177137 |
| electricity | 720 | phase6_asx_individual_revin | 181 | 0.17811 |
| electricity | 720 | phase6_asx_period_multi | 181 | 0.174302 |
| electricity | 96 | phase6_asx_cross | 181 | 0.119219 |
| electricity | 96 | phase6_asx_cross_clip05 | 181 | 0.119289 |
| electricity | 96 | phase6_asx_individual | 181 | 0.11977 |
| electricity | 96 | phase6_asx_individual_period | 181 | 0.11868 |
| electricity | 96 | phase6_asx_individual_revin | 181 | 0.119742 |
| electricity | 96 | phase6_asx_period_multi | 181 | 0.116401 |
| traffic | 192 | phase6_asx_cross | 181 | 0.335048 |
| traffic | 192 | phase6_asx_cross_clip05 | 181 | 0.334217 |
| traffic | 192 | phase6_asx_individual | 181 | 0.347328 |
| traffic | 192 | phase6_asx_individual_period | 181 | 0.346323 |
| traffic | 192 | phase6_asx_individual_revin | 181 | 0.347856 |
| traffic | 192 | phase6_asx_period_multi | 181 | 0.332126 |
| traffic | 336 | phase6_asx_cross | 181 | 0.34709 |
| traffic | 336 | phase6_asx_cross_clip05 | 181 | 0.347172 |
| traffic | 336 | phase6_asx_individual | 181 | 0.358041 |
| traffic | 336 | phase6_asx_individual_period | 181 | 0.35677 |
| traffic | 336 | phase6_asx_individual_revin | 181 | 0.359312 |
| traffic | 336 | phase6_asx_period_multi | 181 | 0.343343 |
| traffic | 720 | phase6_asx_cross | 181 | 0.395136 |
| traffic | 720 | phase6_asx_cross_clip05 | 181 | 0.395733 |
| traffic | 720 | phase6_asx_individual | 181 | 0.406701 |
| traffic | 720 | phase6_asx_individual_period | 181 | 0.404594 |
| traffic | 720 | phase6_asx_individual_revin | 181 | 0.408481 |
| traffic | 720 | phase6_asx_period_multi | 181 | 0.386709 |
| traffic | 96 | phase6_asx_cross | 181 | 0.329335 |
| traffic | 96 | phase6_asx_cross_clip05 | 181 | 0.329242 |
| traffic | 96 | phase6_asx_individual | 181 | 0.346612 |
| traffic | 96 | phase6_asx_individual_period | 181 | 0.34586 |
| traffic | 96 | phase6_asx_individual_revin | 181 | 0.346855 |
| traffic | 96 | phase6_asx_period_multi | 181 | 0.328253 |
| weather | 192 | phase6_asx_cross | 31 | 0.437878 |
| weather | 192 | phase6_asx_cross_clip05 | 31 | 0.443234 |
| weather | 192 | phase6_asx_individual | 31 | 0.438676 |
| weather | 192 | phase6_asx_individual_period | 31 | 0.438569 |
| weather | 192 | phase6_asx_individual_revin | 31 | 0.437773 |
| weather | 192 | phase6_asx_period_multi | 31 | 0.438096 |
| weather | 336 | phase6_asx_cross | 31 | 0.499075 |
| weather | 336 | phase6_asx_cross_clip05 | 31 | 0.504605 |
| weather | 336 | phase6_asx_individual | 31 | 0.503298 |
| weather | 336 | phase6_asx_individual_period | 31 | 0.503263 |
| weather | 336 | phase6_asx_individual_revin | 31 | 0.502111 |
| weather | 336 | phase6_asx_period_multi | 31 | 0.499061 |
| weather | 720 | phase6_asx_cross | 31 | 0.591443 |
| weather | 720 | phase6_asx_cross_clip05 | 31 | 0.59686 |
| weather | 720 | phase6_asx_individual | 31 | 0.595775 |
| weather | 720 | phase6_asx_individual_period | 31 | 0.595722 |
| weather | 720 | phase6_asx_individual_revin | 31 | 0.596845 |
| weather | 720 | phase6_asx_period_multi | 31 | 0.591314 |
| weather | 96 | phase6_asx_cross | 31 | 0.387855 |
| weather | 96 | phase6_asx_cross_clip05 | 31 | 0.38526 |
| weather | 96 | phase6_asx_individual | 31 | 0.382135 |
| weather | 96 | phase6_asx_individual_period | 31 | 0.382017 |
| weather | 96 | phase6_asx_individual_revin | 31 | 0.381475 |
| weather | 96 | phase6_asx_period_multi | 31 | 0.387529 |

## Best cut_freq by test (ANALYSIS ONLY -- not for selection)

| dataset | pred_len | arm | best_cut_freq_test | test_mse |
| --- | ---: | --- | ---: | ---: |
| ETTh1 | 192 | phase6_asx_cross | 181 | 0.418848 |
| ETTh1 | 192 | phase6_asx_cross_clip05 | 181 | 0.414458 |
| ETTh1 | 192 | phase6_asx_individual | 181 | 0.432499 |
| ETTh1 | 192 | phase6_asx_individual_period | 181 | 0.430397 |
| ETTh1 | 192 | phase6_asx_individual_revin | 181 | 0.436349 |
| ETTh1 | 192 | phase6_asx_period_multi | 181 | 0.416658 |
| ETTh1 | 336 | phase6_asx_cross | 181 | 0.44444 |
| ETTh1 | 336 | phase6_asx_cross_clip05 | 181 | 0.433757 |
| ETTh1 | 336 | phase6_asx_individual | 181 | 0.454942 |
| ETTh1 | 336 | phase6_asx_individual_period | 181 | 0.450916 |
| ETTh1 | 336 | phase6_asx_individual_revin | 181 | 0.466571 |
| ETTh1 | 336 | phase6_asx_period_multi | 181 | 0.442249 |
| ETTh1 | 720 | phase6_asx_cross | 181 | 0.454626 |
| ETTh1 | 720 | phase6_asx_cross_clip05 | 181 | 0.431043 |
| ETTh1 | 720 | phase6_asx_individual | 181 | 0.456004 |
| ETTh1 | 720 | phase6_asx_individual_period | 181 | 0.45213 |
| ETTh1 | 720 | phase6_asx_individual_revin | 25 | 0.481158 |
| ETTh1 | 720 | phase6_asx_period_multi | 181 | 0.448751 |
| ETTh1 | 96 | phase6_asx_cross | 181 | 0.382848 |
| ETTh1 | 96 | phase6_asx_cross_clip05 | 181 | 0.379997 |
| ETTh1 | 96 | phase6_asx_individual | 25 | 0.390342 |
| ETTh1 | 96 | phase6_asx_individual_period | 25 | 0.389594 |
| ETTh1 | 96 | phase6_asx_individual_revin | 25 | 0.390026 |
| ETTh1 | 96 | phase6_asx_period_multi | 181 | 0.381765 |
| ETTm1 | 192 | phase6_asx_cross | 43 | 0.349505 |
| ETTm1 | 192 | phase6_asx_cross_clip05 | 43 | 0.353204 |
| ETTm1 | 192 | phase6_asx_individual | 43 | 0.345064 |
| ETTm1 | 192 | phase6_asx_individual_period | 43 | 0.343985 |
| ETTm1 | 192 | phase6_asx_individual_revin | 43 | 0.344973 |
| ETTm1 | 192 | phase6_asx_period_multi | 43 | 0.346639 |
| ETTm1 | 336 | phase6_asx_cross | 43 | 0.379319 |
| ETTm1 | 336 | phase6_asx_cross_clip05 | 43 | 0.378102 |
| ETTm1 | 336 | phase6_asx_individual | 43 | 0.376329 |
| ETTm1 | 336 | phase6_asx_individual_period | 43 | 0.375191 |
| ETTm1 | 336 | phase6_asx_individual_revin | 43 | 0.376144 |
| ETTm1 | 336 | phase6_asx_period_multi | 43 | 0.37667 |
| ETTm1 | 720 | phase6_asx_cross | 43 | 0.424793 |
| ETTm1 | 720 | phase6_asx_cross_clip05 | 43 | 0.42411 |
| ETTm1 | 720 | phase6_asx_individual | 43 | 0.426381 |
| ETTm1 | 720 | phase6_asx_individual_period | 43 | 0.425106 |
| ETTm1 | 720 | phase6_asx_individual_revin | 43 | 0.426514 |
| ETTm1 | 720 | phase6_asx_period_multi | 43 | 0.423044 |
| ETTm1 | 96 | phase6_asx_cross | 43 | 0.318785 |
| ETTm1 | 96 | phase6_asx_cross_clip05 | 43 | 0.31996 |
| ETTm1 | 96 | phase6_asx_individual | 43 | 0.30783 |
| ETTm1 | 96 | phase6_asx_individual_period | 43 | 0.307015 |
| ETTm1 | 96 | phase6_asx_individual_revin | 43 | 0.307806 |
| ETTm1 | 96 | phase6_asx_period_multi | 43 | 0.316857 |
| PEMS04 | 12 | phase6_asx_cross | 49 | 0.093211 |
| PEMS04 | 12 | phase6_asx_cross_clip05 | 49 | 0.0956966 |
| PEMS04 | 12 | phase6_asx_individual | 49 | 0.121114 |
| PEMS04 | 12 | phase6_asx_individual_period | 49 | 0.120684 |
| PEMS04 | 12 | phase6_asx_individual_revin | 49 | 0.121113 |
| PEMS04 | 12 | phase6_asx_period_multi | 49 | 0.0931334 |
| PEMS04 | 24 | phase6_asx_cross | 49 | 0.139606 |
| PEMS04 | 24 | phase6_asx_cross_clip05 | 49 | 0.15063 |
| PEMS04 | 24 | phase6_asx_individual | 49 | 0.225591 |
| PEMS04 | 24 | phase6_asx_individual_period | 49 | 0.224256 |
| PEMS04 | 24 | phase6_asx_individual_revin | 49 | 0.225588 |
| PEMS04 | 24 | phase6_asx_period_multi | 49 | 0.138738 |
| PEMS04 | 48 | phase6_asx_cross | 49 | 0.251873 |
| PEMS04 | 48 | phase6_asx_cross_clip05 | 49 | 0.290361 |
| PEMS04 | 48 | phase6_asx_individual | 49 | 0.505913 |
| PEMS04 | 48 | phase6_asx_individual_period | 49 | 0.502697 |
| PEMS04 | 48 | phase6_asx_individual_revin | 49 | 0.505883 |
| PEMS04 | 48 | phase6_asx_period_multi | 49 | 0.249868 |
| PEMS04 | 96 | phase6_asx_cross | 49 | 0.393524 |
| PEMS04 | 96 | phase6_asx_cross_clip05 | 49 | 0.612718 |
| PEMS04 | 96 | phase6_asx_individual | 49 | 1.01435 |
| PEMS04 | 96 | phase6_asx_individual_period | 49 | 1.00813 |
| PEMS04 | 96 | phase6_asx_individual_revin | 49 | 1.01151 |
| PEMS04 | 96 | phase6_asx_period_multi | 49 | 0.381494 |
| PEMS08 | 12 | phase6_asx_cross | 49 | 0.0921295 |
| PEMS08 | 12 | phase6_asx_cross_clip05 | 49 | 0.0947272 |
| PEMS08 | 12 | phase6_asx_individual | 49 | 0.117777 |
| PEMS08 | 12 | phase6_asx_individual_period | 49 | 0.117358 |
| PEMS08 | 12 | phase6_asx_individual_revin | 49 | 0.117777 |
| PEMS08 | 12 | phase6_asx_period_multi | 49 | 0.0914837 |
| PEMS08 | 24 | phase6_asx_cross | 49 | 0.150745 |
| PEMS08 | 24 | phase6_asx_cross_clip05 | 49 | 0.158126 |
| PEMS08 | 24 | phase6_asx_individual | 49 | 0.223927 |
| PEMS08 | 24 | phase6_asx_individual_period | 49 | 0.222632 |
| PEMS08 | 24 | phase6_asx_individual_revin | 49 | 0.223925 |
| PEMS08 | 24 | phase6_asx_period_multi | 49 | 0.149582 |
| PEMS08 | 48 | phase6_asx_cross | 49 | 0.301337 |
| PEMS08 | 48 | phase6_asx_cross_clip05 | 49 | 0.324344 |
| PEMS08 | 48 | phase6_asx_individual | 49 | 0.522667 |
| PEMS08 | 48 | phase6_asx_individual_period | 49 | 0.519624 |
| PEMS08 | 48 | phase6_asx_individual_revin | 49 | 0.522536 |
| PEMS08 | 48 | phase6_asx_period_multi | 49 | 0.29149 |
| PEMS08 | 96 | phase6_asx_cross | 49 | 0.579403 |
| PEMS08 | 96 | phase6_asx_cross_clip05 | 49 | 0.717465 |
| PEMS08 | 96 | phase6_asx_individual | 49 | 1.11728 |
| PEMS08 | 96 | phase6_asx_individual_period | 49 | 1.11244 |
| PEMS08 | 96 | phase6_asx_individual_revin | 49 | 1.11243 |
| PEMS08 | 96 | phase6_asx_period_multi | 49 | 0.561115 |
| electricity | 192 | phase6_asx_cross | 181 | 0.156384 |
| electricity | 192 | phase6_asx_cross_clip05 | 181 | 0.155345 |
| electricity | 192 | phase6_asx_individual | 181 | 0.15903 |
| electricity | 192 | phase6_asx_individual_period | 181 | 0.157527 |
| electricity | 192 | phase6_asx_individual_revin | 181 | 0.15915 |
| electricity | 192 | phase6_asx_period_multi | 181 | 0.151979 |
| electricity | 336 | phase6_asx_cross | 181 | 0.171584 |
| electricity | 336 | phase6_asx_cross_clip05 | 181 | 0.171554 |
| electricity | 336 | phase6_asx_individual | 181 | 0.173817 |
| electricity | 336 | phase6_asx_individual_period | 181 | 0.171962 |
| electricity | 336 | phase6_asx_individual_revin | 181 | 0.174186 |
| electricity | 336 | phase6_asx_period_multi | 181 | 0.166828 |
| electricity | 720 | phase6_asx_cross | 181 | 0.209717 |
| electricity | 720 | phase6_asx_cross_clip05 | 181 | 0.206815 |
| electricity | 720 | phase6_asx_individual | 181 | 0.209144 |
| electricity | 720 | phase6_asx_individual_period | 181 | 0.206207 |
| electricity | 720 | phase6_asx_individual_revin | 181 | 0.209677 |
| electricity | 720 | phase6_asx_period_multi | 181 | 0.203556 |
| electricity | 96 | phase6_asx_cross | 181 | 0.141479 |
| electricity | 96 | phase6_asx_cross_clip05 | 181 | 0.142084 |
| electricity | 96 | phase6_asx_individual | 181 | 0.142635 |
| electricity | 96 | phase6_asx_individual_period | 181 | 0.141463 |
| electricity | 96 | phase6_asx_individual_revin | 181 | 0.142674 |
| electricity | 96 | phase6_asx_period_multi | 181 | 0.138265 |
| traffic | 192 | phase6_asx_cross | 181 | 0.405761 |
| traffic | 192 | phase6_asx_cross_clip05 | 181 | 0.403841 |
| traffic | 192 | phase6_asx_individual | 181 | 0.425683 |
| traffic | 192 | phase6_asx_individual_period | 181 | 0.424449 |
| traffic | 192 | phase6_asx_individual_revin | 181 | 0.425921 |
| traffic | 192 | phase6_asx_period_multi | 181 | 0.400557 |
| traffic | 336 | phase6_asx_cross | 181 | 0.41926 |
| traffic | 336 | phase6_asx_cross_clip05 | 181 | 0.417442 |
| traffic | 336 | phase6_asx_individual | 181 | 0.438215 |
| traffic | 336 | phase6_asx_individual_period | 181 | 0.43679 |
| traffic | 336 | phase6_asx_individual_revin | 181 | 0.438476 |
| traffic | 336 | phase6_asx_period_multi | 181 | 0.411927 |
| traffic | 720 | phase6_asx_cross | 181 | 0.472758 |
| traffic | 720 | phase6_asx_cross_clip05 | 181 | 0.457618 |
| traffic | 720 | phase6_asx_individual | 181 | 0.476338 |
| traffic | 720 | phase6_asx_individual_period | 181 | 0.474066 |
| traffic | 720 | phase6_asx_individual_revin | 181 | 0.476954 |
| traffic | 720 | phase6_asx_period_multi | 181 | 0.448077 |
| traffic | 96 | phase6_asx_cross | 181 | 0.390343 |
| traffic | 96 | phase6_asx_cross_clip05 | 181 | 0.390999 |
| traffic | 96 | phase6_asx_individual | 181 | 0.416057 |
| traffic | 96 | phase6_asx_individual_period | 181 | 0.415094 |
| traffic | 96 | phase6_asx_individual_revin | 181 | 0.416248 |
| traffic | 96 | phase6_asx_period_multi | 181 | 0.388673 |
| weather | 192 | phase6_asx_cross | 31 | 0.197634 |
| weather | 192 | phase6_asx_cross_clip05 | 31 | 0.203098 |
| weather | 192 | phase6_asx_individual | 31 | 0.190698 |
| weather | 192 | phase6_asx_individual_period | 31 | 0.190687 |
| weather | 192 | phase6_asx_individual_revin | 31 | 0.190203 |
| weather | 192 | phase6_asx_period_multi | 31 | 0.197595 |
| weather | 336 | phase6_asx_cross | 31 | 0.249578 |
| weather | 336 | phase6_asx_cross_clip05 | 31 | 0.252263 |
| weather | 336 | phase6_asx_individual | 31 | 0.240569 |
| weather | 336 | phase6_asx_individual_period | 31 | 0.240582 |
| weather | 336 | phase6_asx_individual_revin | 31 | 0.240016 |
| weather | 336 | phase6_asx_period_multi | 31 | 0.249379 |
| weather | 720 | phase6_asx_cross | 31 | 0.312987 |
| weather | 720 | phase6_asx_cross_clip05 | 31 | 0.318011 |
| weather | 720 | phase6_asx_individual | 31 | 0.309688 |
| weather | 720 | phase6_asx_individual_period | 31 | 0.309708 |
| weather | 720 | phase6_asx_individual_revin | 31 | 0.31011 |
| weather | 720 | phase6_asx_period_multi | 31 | 0.313159 |
| weather | 96 | phase6_asx_cross | 31 | 0.152079 |
| weather | 96 | phase6_asx_cross_clip05 | 31 | 0.157925 |
| weather | 96 | phase6_asx_individual | 31 | 0.148135 |
| weather | 96 | phase6_asx_individual_period | 31 | 0.148103 |
| weather | 96 | phase6_asx_individual_revin | 31 | 0.147851 |
| weather | 96 | phase6_asx_period_multi | 31 | 0.151885 |

