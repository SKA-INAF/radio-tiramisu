
import wandb
import datetime

class Logger:
    def __init__(self, out_file, classes, writer):
        self.out_file = out_file
        self.classes = classes[1:]
        self.writer = writer
        self.metric_names = ['accuracy', 'iou', 'precision', 'recall', 'dice', 'obj_precision', 'obj_recall']
        with open(out_file, 'w') as out:
            out.write('Date: ' + str(datetime.datetime.now()) + '\n')

    def log_metrics(self, split, epoch, loss, metrics, time_elapsed):

        # Tensorboard
        self.writer.add_scalar(f"Loss/{split.lower()}", loss, epoch)
        for class_name in self.classes:
            for metric_name in metrics[class_name]:
                self.writer.add_scalar(metric_name + f"/{split.lower()}_" + class_name, metrics[class_name][metric_name], epoch)
        
        # Header
        print(f'Epoch {epoch}\n{split} - Loss: {loss:.4f}'.format(epoch, loss))
        print('Per class metrics: ')

        with open(self.out_file, 'a') as out:
            out.write(f'Epoch {epoch}\n{split} - Loss: {loss:.4f}\n')
            out.write('Per class metrics: \n')
        
        # Metrics
        for i, class_name in enumerate(self.classes):
            print('\t {}: \tAcc: {:.4f}, \tIoU: {:.4f}, \tSensitivity: {:.4f}, \tPrecision: {:.4f}, \tDice: {:.4f}, \tObject Precision: {:.4f}, \tObject Recall: {:.4f}'.format(class_name, metrics[class_name]['accuracy'], metrics[class_name]['iou'], metrics[class_name]['recall'], metrics[class_name]['precision'], metrics[class_name]['dice'], metrics[class_name]['obj_precision'], metrics[class_name]['obj_recall']))
            with open(self.out_file, 'a') as out:
                out.write('\t {}: \tAcc: {:.4f}, \tIoU: {:.4f}, \tSensitivity: {:.4f}, \tPrecision: {:.4f}, \tDice: {:.4f}, \tObject Precision: {:.4f}, \tObject Recall: {:.4f}\n'.format(class_name, metrics[class_name]['accuracy'], metrics[class_name]['iou'], metrics[class_name]['recall'], metrics[class_name]['precision'], metrics[class_name]['dice'], metrics[class_name]['obj_precision'], metrics[class_name]['obj_recall']))

        # Time elapsed
        print(f'{split} Time {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        with open(self.out_file, 'a') as out:
            out.write(f'{split} Time {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s\n')


    def log_header(self, n_samples, inputs_size, targets_size):
        for split in n_samples.keys():
            print(f'{split}: {n_samples[split]} samples')

        print("Inputs shape: ", inputs_size)
        print("Targets shape: ", targets_size)

    def wandb_plot_metrics(self, metrics, split):
        for class_name in self.classes[1:]:
            wandb.log({split + '/' + class_name + '/' + metric_name: metrics[class_name][metric_name] for metric_name in metrics[class_name]})