from federatedscope.register import register_worker
from federatedscope.core.workers import Server, Client
from federatedscope.core.message import Message
import logging
import torch
from federatedscope.core.auxiliaries.utils import merge_dict_of_results, \
    Timeout, merge_param_dict
import torch

logger = logging.getLogger(__name__)


# Build your worker here.
class POIV4Server(Server):
    def check_and_move_on(self,
                          check_eval_result=False,
                          min_received_num=None):
        # TODO: 需要完善当采样率不等于0时的实现
        min_received_num = len(self.comm_manager.get_neighbors().keys())

        if check_eval_result and self._cfg.federate.mode.lower(
        ) == "standalone":
            # in evaluation stage and standalone simulation mode, we assume
            # strong synchronization that receives responses from all clients
            min_received_num = len(self.comm_manager.get_neighbors().keys())

        move_on_flag = True

        # round or finishing the evaluation
        if self.check_buffer(self.state, min_received_num, check_eval_result):
            if not check_eval_result:
                # Receiving enough feedback in the training process
                #################################################################
                local_protos = dict()
                msg_list = self.msg_buffer['train'][self.state]
                for key, values in msg_list.items():
                    local_protos[key] = values[1]
                #################################################################

                self.state += 1
                if self.state % self._cfg.eval.freq == 0 and self.state != \
                        self.total_round_num:
                    #  Evaluate
                    logger.info(f'Server: Starting evaluation at the end '
                                f'of round {self.state - 1}.')
                    self.eval()

                if self.state < self.total_round_num:
                    # Move to next round of training
                    logger.info(
                        f'----------- Starting a new training round (Round '
                        f'#{self.state}) -------------')
                    # Clean the msg_buffer
                    self.msg_buffer['train'][self.state - 1].clear()
                    self.msg_buffer['train'][self.state] = dict()
                    self.staled_msg_buffer.clear()
                    # Start a new training round
                    self._broadcast_custom_message(msg_type='all_local_protos', content=local_protos)
                else:
                    # Final Evaluate
                    logger.info('Server: Training is finished! Starting '
                                'evaluation.')
                    self.eval()

            else:
                # Receiving enough feedback in the evaluation process
                self._merge_and_format_eval_results()
                if self.state >= self.total_round_num:
                    self.is_finish = True

        else:
            move_on_flag = False

        return move_on_flag

    def eval(self):
        self._broadcast_custom_message(msg_type='evaluate', content=None, filter_unseen_clients=False)

    def _broadcast_custom_message(self, msg_type, content,
                                  sample_client_num=-1,
                                  filter_unseen_clients=True):
        if filter_unseen_clients:
            # to filter out the unseen clients when sampling
            self.sampler.change_state(self.unseen_clients_id, 'unseen')

        if sample_client_num > 0:
            receiver = self.sampler.sample(size=sample_client_num)
        else:
            # broadcast to all clients
            receiver = list(self.comm_manager.neighbors.keys())
            if msg_type == 'model_para':
                self.sampler.change_state(receiver, 'working')

        rnd = self.state - 1 if msg_type == 'evaluate' else self.state

        self.comm_manager.send(
            Message(msg_type=msg_type,
                    sender=self.ID,
                    receiver=receiver,
                    state=min(rnd, self.total_round_num),
                    timestamp=self.cur_timestamp,
                    content=content))

        if filter_unseen_clients:
            # restore the state of the unseen clients within sampler
            self.sampler.change_state(self.unseen_clients_id, 'seen')


class POIV4Client(Client):
    def __init__(self,
                 ID=-1,
                 server_id=None,
                 state=-1,
                 config=None,
                 data=None,
                 model=None,
                 device='cpu',
                 strategy=None,
                 is_unseen_client=False,
                 *args,
                 **kwargs):
        super(POIV4Client, self).__init__(ID, server_id, state, config, data, model, device,
                                          strategy, is_unseen_client, *args, **kwargs)
        self.trainer.ctx.local_protos_from_other = []
        self.trainer.ctx.client_ID = self.ID
        self.register_handlers('all_local_protos', self.callback_funcs_for_model_para, ['model_para', 'ss_model_para'])

        # For visualization of node embedding
        self.client_agg_proto = dict()
        self.client_node_emb_all = dict()
        self.client_node_labels = dict()
        self.glob_proto_on_client = dict()
        self.client_PL_node_emb_all = dict()

    def callback_funcs_for_model_para(self, message: Message):
        round = message.state
        sender = message.sender
        timestamp = message.timestamp
        content = message.content

        if message.msg_type == 'all_local_protos':
            self.trainer.update(content)
        self.state = round
        self.trainer.ctx.cur_state = self.state
        sample_size, model_para, results, agg_protos = self.trainer.train()

        train_log_res = self._monitor.format_eval_res(
            results,
            rnd=self.state,
            role='Client #{}'.format(self.ID),
            return_raw=True)
        logger.info(train_log_res)

        if self._cfg.wandb.use and self._cfg.wandb.client_train_info:
            self._monitor.save_formatted_results(train_log_res,
                                                 save_file_name="")

        self.history_results = merge_dict_of_results(
            self.history_results, train_log_res['Results_raw'])

        if self._cfg.vis_embedding:
            self.glob_proto_on_client[round] = self.trainer.ctx.global_protos
            self.client_node_emb_all[round] = self.trainer.ctx.node_emb_all
            self.client_node_labels[round] = self.trainer.ctx.node_labels
            self.client_agg_proto[round] = agg_protos
            self.client_PL_node_emb_all[round] = self.trainer.ctx.PL_node_emb_all

        self.comm_manager.send(
            Message(msg_type='model_para',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    content=(sample_size, agg_protos)))

    def callback_funcs_for_evaluate(self, message: Message):
        """
        The handling function for receiving the request of evaluating

        Arguments:
            message: The received message
        """
        sender, timestamp = message.sender, message.timestamp
        self.state = message.state
        if message.content is not None and self._cfg.federate.method not in ['fedmd']:  # TODO:检查fedmd是否会更新模型
            self.trainer.update(message.content,
                                strict=self._cfg.federate.share_local_model)
        if self.early_stopper.early_stopped and self._cfg.federate.method in [
            "local", "global"
        ]:
            metrics = list(self.best_results.values())[0]
        else:
            metrics = {}
            if self._cfg.finetune.before_eval:
                self.trainer.finetune()
            for split in self._cfg.eval.split:
                # TODO: The time cost of evaluation is not considered here
                eval_metrics = self.trainer.evaluate(
                    target_data_split_name=split)

                if self._cfg.federate.mode == 'distributed':
                    logger.info(
                        self._monitor.format_eval_res(eval_metrics,
                                                      rnd=self.state,
                                                      role='Client #{}'.format(
                                                          self.ID),
                                                      return_raw=True))

                metrics.update(**eval_metrics)

            formatted_eval_res = self._monitor.format_eval_res(
                metrics,
                rnd=self.state,
                role='Client #{}'.format(self.ID),
                forms=['raw'],
                return_raw=True)
            self._monitor.update_best_result(self.best_results,
                                             formatted_eval_res['Results_raw'],
                                             results_type=f"client #{self.ID}")
            self.history_results = merge_dict_of_results(
                self.history_results, formatted_eval_res['Results_raw'])
            self.early_stopper.track_and_check(self.history_results[
                                                   self._cfg.eval.best_res_update_round_wise_key])

            if self._cfg.wandb.use and self._cfg.wandb.client_train_info:
                self._monitor.save_formatted_results(formatted_eval_res['Results_raw'],
                                                     save_file_name="")

        self.comm_manager.send(
            Message(msg_type='metrics',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    timestamp=timestamp,
                    content=metrics))

    def callback_funcs_for_finish(self, message: Message):
        logger.info(
            f"================= client {self.ID} received finish message "
            f"=================")

        if message.content is not None:
            self.trainer.update(message.content, strict=True)
        if self._cfg.vis_embedding:
            folderPath = self._cfg.MHFL.emb_file_path
            torch.save(self.glob_proto_on_client, f'{folderPath}/global_protos_on_client_{self.ID}.pth')  # 全局原型
            torch.save(self.client_agg_proto, f'{folderPath}/agg_protos_on_client_{self.ID}.pth')  # 本地原型
            torch.save(self.client_node_emb_all,
                       f'{folderPath}/local_node_embdeddings_on_client_{self.ID}.pth')  # 每个节点的embedding
            torch.save(self.client_node_labels, f'{folderPath}/node_labels_on_client_{self.ID}.pth')  # 标签
            torch.save(self.data, f'{folderPath}/raw_data_on_client_{self.ID}.pth')  # 划分给这个client的pyg data
            torch.save(self.client_PL_node_emb_all, f'{folderPath}/PP_node_embeddings_on_client_{self.ID}.pth')
        self._monitor.finish_fl()


def call_my_worker(method):
    if method == 'poiv4':
        worker_builder = {'client': POIV4Client, 'server': POIV4Server}
        return worker_builder


register_worker('poiv4', call_my_worker)
