import utils
import random
import dataset
import numpy as np
import logging
import os
import json
import time
import csv
from tqdm import tqdm
import warnings
import math
from models.mamba import Mamba, ModelArgs
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from sklearn.model_selection import train_test_split
from models import testmamba as testmamba
from models.testmamba import MambaTextClassification


from transformers import (
    AdamW,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


task_metrics = {"steganalysis" : "accuracy",
                "graph_steganalysis" : "accuracy",}


logger = logging.getLogger(__name__)
time_stamp = "-".join(time.ctime().split())
from transformers import logging as lg2
lg2.set_verbosity_error()
warnings.filterwarnings('ignore')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(Configs, VOCAB_SIZE=None, checkpoint=None):
    # set model

    logger.info("----------------init model-----------------------")


    # if Configs.model.lower() in ["birnn"]:
    #     Model_Configs = Configs.RNN
    #     model = BiRNN.BERT_TC.from_pretrained(model_name_or_path,
    #                                         **{**Model_Configs, "class_num": Configs.class_num, })

    if checkpoint is not None:
        logger.info("---------------------loading model from {}------------\n\n".format(checkpoint))
        model_name_or_path = checkpoint
    else:
        logger.info("-------------loading pretrained language model from huggingface--------------\n\n")
        model_name_or_path = Configs.model_name_or_path


    if Configs.model.lower() in ["mamba"]:
        Model_Configs = Configs.RNN
        model = MambaTextClassification.from_pretrained(model_name_or_path)
        model.to("cuda")

    elif Configs.model.lower() in ["testmamba"]:
        Model_Configs = Configs.RNN
        model = MambaTextClassification.from_pretrained(model_name_or_path)
        model.to("cuda")

    else:
        logger.error("no such model, exit")
        exit()


    logger.info("Model Configs")
    logger.info(json.dumps({**{"MODEL_TYPE": Configs.model}, **Model_Configs, }))
    model = model.to(Configs.device)
    return model

def train( model, Configs, tokenizer):

    train_dataset = load_and_cache_examples(Configs.Dataset, Configs.task_name, tokenizer)
    Training_Configs = Configs.Training_with_Processor
    Training_Configs.train_batch_size = Training_Configs.per_gpu_train_batch_size * max(1, Training_Configs.n_gpu)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=Training_Configs.train_batch_size)

    t_total = len(train_dataloader) // Training_Configs.gradient_accumulation_steps * Training_Configs.num_train_epochs

    num_warmup_steps = int(Training_Configs.warmup_ratio * t_total)
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": Training_Configs.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=Training_Configs.learning_rate, eps=Training_Configs.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(Training_Configs.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(Training_Configs.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(Training_Configs.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(Training_Configs.model_name_or_path, "scheduler.pt")))


    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", Training_Configs.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", Training_Configs.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        Training_Configs.train_batch_size
        * Training_Configs.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", Training_Configs.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = range(epochs_trained, int(Training_Configs.num_train_epochs))

    set_seed(Configs.seed)  # Added here for reproductibility

    best_val_metric = None
    for epoch_n in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch_n}", disable=False)
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(Configs.device) for t in batch)

            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2], "labels": batch[3]}
            if Configs.task_name == "graph_steganalysis":
                inputs = {**inputs,"graph":batch[4]}

            outputs = model(**inputs)
            loss = outputs[0]
            loss.backward()

            new_loss, new_logits = outputs[:2]
            new_preds = new_logits.detach().cpu().numpy()
            new_out_label_ids = inputs["labels"].detach().cpu().numpy()
            new_preds = np.argmax(new_preds, axis=1)
            different_elements = 0
            different_elements = np.sum(new_preds != new_out_label_ids)

            tr_loss += loss.item()
            if (step + 1) % Training_Configs.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), Training_Configs.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                logs = {}

                if ( Training_Configs.eval_and_save_steps > 0 and global_step % Training_Configs.eval_and_save_steps == 0) \
                        or (step+1==t_total):
                    print("evaluate evaluate evaluate evaluate evaluate evaluate")
                    results, _, _ = evaluate(model, tokenizer, Configs, Configs.task_name, use_tqdm=False)
                    # logger.info("------Next Evalset will be loaded from cached file------")
                    Configs.Dataset.overwrite_cache = False
                    for key, value in results.items():
                        logs[f"eval_{key}"] = value
                    logger.info(json.dumps({**logs, **{"step": global_step}}))

                    # save
                    if Training_Configs.save_only_best:
                        output_dirs = [os.path.join(Configs.out_dir, Configs.checkpoint)]
                    else:
                        output_dirs = [os.path.join(Configs.out_dir, f"checkpoint-{global_step}")]
                    curr_val_metric = results[task_metrics[Configs.task_name]]
                    if best_val_metric is None or curr_val_metric > best_val_metric:
                        # check if best model so far
                        logger.info("Congratulations, best model so far!")
                        best_val_metric = curr_val_metric

                        for output_dir in output_dirs:
                            # in each dir, save model, tokenizer, args, optimizer, scheduler
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            model_to_save = (
                                model.module if hasattr(model, "module") else model
                            )  # Take care of distributed/parallel training
                            logger.info("Saving model checkpoint to %s", output_dir)
                            if Configs.use_plm:
                                model_to_save.save_pretrained(output_dir)
                            else:
                                torch.save(model_to_save, os.path.join(output_dir, "pytorch_model.bin"))
                            torch.save(Configs.state_dict, os.path.join(output_dir, "training_args.bin"))
                            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                            tokenizer.save_pretrained(output_dir)
                            logger.info("\tSaved model checkpoint to %s", output_dir)


            # if Training_Configs.max_steps > 0 and global_step > Training_Configs.max_steps:
            #     epoch_iterator.close()
            #     break
        if Training_Configs.max_steps > 0 and global_step > Training_Configs.max_steps:
            # train_iterator.close()
            break

    return global_step, tr_loss / global_step


def evaluate(model, tokenizer, Configs, task_name, split="dev", prefix="", use_tqdm=True):
    Training_Configs = Configs.Training_with_Processor
    results = {}

    eval_dataset = load_and_cache_examples(Configs.Dataset, task_name, tokenizer, split=split)

    if not os.path.exists(Configs.out_dir):
        os.makedirs(Configs.out_dir)

    Training_Configs.eval_batch_size = Training_Configs.per_gpu_eval_batch_size * max(1, Training_Configs.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=Training_Configs.eval_batch_size)

    # # multi-gpu eval
    # if Training_Configs.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
    #     model = torch.nn.DataParallel(model)

    # Eval!
    logger.info(f"***** Running evaluation: {prefix} on {task_name} {split} *****")
    logger.info("Num examples = %d", len(eval_dataset))
    logger.info("Batch size = %d", Training_Configs.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    ex_ids = None
    eval_dataloader = tqdm(eval_dataloader, desc="Evaluating") if use_tqdm else eval_dataloader
    for batch in eval_dataloader:
        model.eval()
        batch = tuple(t.to(Configs.device) for t in batch)
        guids = batch[-1]

        max_seq_length = batch[0].size(1)
        if Training_Configs.use_fixed_seq_length:  # no dynamic sequence length
            batch_seq_length = max_seq_length
        else:
            batch_seq_length = torch.max(batch[-2], 0)[0].item()

        if batch_seq_length < max_seq_length:
            inputs = {"input_ids": batch[0][:, :batch_seq_length].contiguous(),
                      "attention_mask": batch[1][:, :batch_seq_length].contiguous(),
                      "token_type_ids":batch[2][:, :batch_seq_length].contiguous(),
                      "labels": batch[3]}

        else:
            inputs = {"input_ids": batch[0], "attention_mask": batch[1],  "token_type_ids":batch[2],"labels": batch[3]}

        if Configs.task_name == "graph_steganalysis":
            inputs = {**inputs,"graph":batch[4]}

        with torch.no_grad():
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            eval_loss += tmp_eval_loss.mean().item()

        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
            ex_ids = [guids.detach().cpu().numpy()]
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
            ex_ids.append(guids.detach().cpu().numpy())

    ex_ids = np.concatenate(ex_ids, axis=0)
    eval_loss = eval_loss / nb_eval_steps

    preds = np.argmax(preds, axis=1)

    result = utils.compute_metrics(task_name, preds, out_label_ids,)
    results.update(result)
    if prefix == "":
        return results, preds, ex_ids
    output_eval_file = os.path.join(Configs.out_dir, prefix, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info(f"***** {split} results: {prefix} *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return results, preds, ex_ids


def load_and_cache_examples(Dataset_Configs, task, tokenizer, split="train"):
    if task == "steganalysis":
        from processors.process import SteganalysisProcessor as DataProcessor
    elif task == "graph_steganalysis":
        from processors.graph_process import GraphSteganalysisProcessor as DataProcessor

    processor  = DataProcessor(tokenizer)
    # Load data features from cache or dataset file
    cached_tensors_file = os.path.join(
        Dataset_Configs.csv_dir,
        "tensors_{}_{}_{}".format(
            split, time_stamp, str(task),
        ),
    )
    if os.path.exists(cached_tensors_file) and not Dataset_Configs.overwrite_cache:
        logger.info("Loading tensors from cached file %s", cached_tensors_file)
        start_time = time.time()
        dataset = torch.load(cached_tensors_file)
    else:
        # no cached tensors, process data from scratch
        logger.info("Creating features from dataset file at %s", Dataset_Configs.csv_dir)
        if split == "train":
            get_examples = processor.get_train_examples
        elif split == "dev":
            get_examples = processor.get_dev_examples
        elif split == "test":
            get_examples = processor.get_test_examples

        examples = get_examples(Dataset_Configs.csv_dir)
        dataset = processor.convert_examples_to_features(examples,)
        logger.info("Finished creating features")

        logger.info("Finished converting features into tensors")
        if Dataset_Configs.save_cache:
            logger.info("Saving features into cached file %s", cached_tensors_file)
            torch.save(dataset, cached_tensors_file)
            logger.info("Finished saving tensors")

    if task == "record" and split in ["dev", "test"]:
        answers = processor.get_answers(Dataset_Configs.csv_dir, split)
        return dataset, answers
    else:
        return dataset


def main(Configs, seed_shift=0):
    # args conflict checking
    if Configs.use_plm:
            assert Configs.use_processor, "\nWhen using plm, You can only use processor to process dataset!!\n"

    Dataset_Configs = Configs.Dataset
    Configs.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(Configs.out_dir,exist_ok=True)
    set_seed(Configs.seed+seed_shift)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    handler = logging.FileHandler(os.path.join(Configs.out_dir,time_stamp+"_log"))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info("----------------use plm or not----------------")
    if Configs.use_plm:
        logger.info("------------------YES-----------------------------")
        Configs.model_name_or_path = Configs.Training_with_Processor.model_name_or_path
        logger.info("\tload plm name or path from Training_with_Processor args")
    else:
        logger.info("--------------------NO-------------------------")
        if Configs.use_processor:
            Configs.model_name_or_path = Configs.Training_with_Processor.model_name_or_path
            logger.info("\tload plm name or path from Training_with_Processor args")
        else:
            Configs.model_name_or_path = Configs.Tokenizer.model_name_or_path
            logger.info("\tload plm name or path from Tokenizer args")

    logger.info("-------------------------------------------------------------------------------------------------------")
    # prepare data
    if Configs.use_processor:
        if (False):
            pass
        else:
            os.makedirs(Dataset_Configs.csv_dir, exist_ok=True)
            with open(Dataset_Configs.cover_file, 'r', encoding='utf-8') as f:
                covers = f.read().split("\n")
            covers = list(filter(lambda x: x not in ['', None], covers))
            random.shuffle(covers)
            with open(Dataset_Configs.stego_file, 'r', encoding='utf-8') as f:
                stegos = f.read().split("\n")
            stegos = list(filter(lambda x: x not in ['', None],  stegos))
            random.shuffle(stegos)
            texts = covers+stegos
            labels = [0]*len(covers) + [1]*len(stegos)
            val_ratio = (1-Dataset_Configs.split_ratio)/Dataset_Configs.split_ratio
            train_texts,test_texts,train_labels,test_labels = train_test_split(texts,labels,train_size=Dataset_Configs.split_ratio)
            train_texts,val_texts, train_labels,val_labels,  = train_test_split(train_texts, train_labels, train_size=1-val_ratio)
            def write2file(X, Y, filename):
                with open(filename, "w", encoding="utf-8", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["text", "label"])
                    for x, y in zip(X, Y):
                        writer.writerow([x, y])
            write2file(train_texts,train_labels, os.path.join(Dataset_Configs.csv_dir,"train.csv"))
            write2file(val_texts, val_labels, os.path.join(Dataset_Configs.csv_dir, "val.csv"))
            write2file(test_texts, test_labels, os.path.join(Dataset_Configs.csv_dir, "test.csv"))
        # tokenizer = AutoTokenizer.from_pretrained(Configs.model_name_or_path,)
        tokenizer = AutoTokenizer.from_pretrained('./model/gpt-neox-20b')
        VOCAB_SIZE = tokenizer.vocab_size
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = load_model(Configs, VOCAB_SIZE=VOCAB_SIZE)

    logger.info("--------------start training--------------------")

    if Configs.use_processor:
        #测试， 里面包括验证
        print("train train train train train")
        global_step, tr_loss = train(model, Configs, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
        Training_Configs = Configs.Training_with_Processor

        checkpoints = [os.path.join(Configs.out_dir, Configs.checkpoint)]
        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            tokenizer = AutoTokenizer.from_pretrained(checkpoint, do_lower_case=Training_Configs.do_lower_case)
            prefix = checkpoint.split("/")[-1]
            model = load_model(Configs, VOCAB_SIZE=tokenizer.vocab_size,checkpoint=checkpoint)
            # 只有这个时候会测试
            print("test test test test test test")
            result, preds, ex_ids = evaluate(model, tokenizer, Configs, Configs.task_name, split="test", prefix=prefix)
            test_acc = result["accuracy"]
            test_precision = result["precision"]
            test_recall = result["recall"]
            test_Fscore = result["f1_score"]

    else:
        test_acc, test_precision, test_recall, test_Fscore = train_with_helper(data_helper,model,Configs)

    record_file = Configs.record_file if Configs.record_file is not None else "record.txt"
    result_path = os.path.join(Configs.out_dir, time_stamp+"----"+record_file)
    with open(result_path, "w", encoding="utf-8") as f:
        f.write("test phase:\naccuracy\t{:.4f}\nprecision\t{:.4f}\nrecall\t{:.4f}\nf1_score\t{:.4f}"
                .format(test_acc*100,test_precision*100,test_recall*100,test_Fscore*100))

    return test_acc, test_precision, test_recall, test_Fscore


if __name__ == '__main__':
    import argparse
    import numpy as np
    parser = argparse.ArgumentParser(description="argument for generation")
    parser.add_argument("--config_path", type=str, default="./configs/experiment/3bit.json")
    args = parser.parse_args()
    Configs = utils.Config(args.config_path).get_configs()
    os.environ["CUDA_VISIBLE_DEVICES"] = Configs.gpuid
    total_test_acc=[]
    total_test_precision=[]
    total_test_recall=[]
    total_test_Fscore=[]
    for i in range(Configs.get("repeat_num", 1)):
        test_acc, test_precision, test_recall, test_Fscore = main(Configs,seed_shift=i)
        total_test_acc.append(test_acc)
        total_test_precision.append(test_precision)
        total_test_recall.append(test_recall)
        total_test_Fscore.append(test_Fscore)


    total_test_acc = sorted(total_test_acc)
    total_test_acc = total_test_acc[1:-1]

    total_test_precision = sorted(total_test_precision)
    total_test_precision = total_test_precision[1:-1]

    total_test_recall = sorted(total_test_recall)
    total_test_recall = total_test_recall[1:-1]

    total_test_Fscore = sorted(total_test_Fscore)
    total_test_Fscore = total_test_Fscore[1:-1]


    message = "Final results\n(repeat times: {}):\naccuracy\t{:.2f}%+{:.2f}%\nprecision\t{:.2f}%+{:.2f}%\nrecall\t{:.2f}%+{:.2f}%\nf1_score\t{:.2f}%+{:.2f}%"\
        .format(Configs.get("repeat_num", 1), np.mean(total_test_acc)*100, np.std(total_test_acc)*100,
                np.mean(total_test_precision)*100, np.std(total_test_precision)*100,
                np.mean(total_test_recall)*100, np.std(total_test_recall)*100,
                np.mean(total_test_Fscore)*100, np.std(total_test_Fscore)*100)
    logger.info(message)
