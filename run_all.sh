desc=${1:-Baseline}
notes=${2:-Test}

sh scripts/multivariate_forecasting/ETT/Testformer_ETTh1.sh $desc $notes &&
sh scripts/multivariate_forecasting/ETT/Testformer_ETTh2.sh $desc $notes &&
sh scripts/multivariate_forecasting/ETT/Testformer_ETTm1.sh $desc $notes &&
sh scripts/multivariate_forecasting/ETT/Testformer_ETTm2.sh $desc $notes

