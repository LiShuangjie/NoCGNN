import sys
sys.path.insert(0, "..")

from BaseLogger import BaseLogger

class ACMPythorchLogger(BaseLogger):
    def __init__(self):
        super().__init__("acm-torch-logger")

    def log_best_result(self, model_info, best_result_info):
        msg = (
            f"Best Result - "
            f"dataset: {model_info['dataset_name']}, "
            f"best acc: {best_result_info['best_result']:.4f}, "
            f"Test Mean: {best_result_info['test_result']:.4f}, "
            f"Test Std:{best_result_info['test_std']:.4f}, "
            f"model {model_info['model']}, "
            f"variant: {model_info['variant']}, "
            f"structure_info: {model_info['structure_info']}, "
            # f"init_layers_X: {model_info['init_layers_X']}, "
            # f"Hidden: {model_info['hidden']}, "
            f"knn,k=:  {model_info['knn']}, "
            f"learning rate {best_result_info['lr']:.4f}, "
            f"weight decay {best_result_info['weight_decay']:.6f}, "
            f"dropout {best_result_info['dropout']:.4f}, "
            f"nhid1 {best_result_info['nhid1']}, "
            f"nhid2 {best_result_info['nhid2']}, "
            f"layers {model_info['layers']}, "
            f"epoch average/runtime average time: "
            f"{best_result_info['runtime_average']:.2f}ms/{best_result_info['runtime_average']:.2f}s"
        )
        self.logger.info(msg)
        self.saveResultInfo(msg)

    # 将信息结果保存到文件中
    def saveResultInfo(self, msg):
        file = open('./result.txt', 'a')
        file.write('{}\n'.format(msg))
        file.close()

    def log_param_tune(
        self,
        model_info,
        curr_split,
        curr_dropout,
        curr_weight_decay,
        curr_lr,
        curr_hidden1,
        curr_hidden2,
        curr_res,
        curr_loss,
    ):
        msg = (
            f"Optimization - "
            f"model: {model_info['model']}, "
            f"dataset: {model_info['dataset_name']}, "
            f"lr: {curr_lr:.5f}, "
            f"weight decay {curr_weight_decay:.5f}, "
            f"dropout {curr_dropout:.4f}, "
            f"nhid1 {curr_hidden1}, "
            # f"nhid2 {curr_hidden1}, "
            f"split {curr_split}, "
            f"Best Test Result: {curr_res:.4f}, "
            f"Training Loss: {curr_loss:.4f}"
        )
        self.logger.info(msg)

    def log_run(self, model_info, run_info):
        msg = (
            f"Run {run_info['split']} Summary - "
            f"layers {model_info['layers']}, "
            f"dataset: {model_info['dataset_name']}, "
            f"best acc: {run_info['best_result']}, "
            f"Test Mean: {run_info['result']:.4f}, "
            f"Test Std: {run_info['std']:.4f}, "
            f"model: {model_info['model']}, "
            f"variant: {model_info['variant']}, "
            # f"init_layers_X: {model_info['init_layers_X']}, "
            f"structure_info: {model_info['structure_info']}, "
            f"learning rate {run_info['lr']:.4f}, "
            f"weight decay {run_info['weight_decay']:.6f}, "
            f"dropout {run_info['dropout']:.4f}, "
            # f"nhid {run_info['nhid']}, "
            f"runtime average time: {run_info['runtime_average']:.2f}s, "
            f"epoch average time: {run_info['epoch_average']:.2f}ms"
        )
        self.logger.info(msg)
