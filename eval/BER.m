function [ber, recall_p, recall_n] = BER(pred_map, gt_map)
% Code Author: Haiyang Mei
% Email: mhy666@mail.dlut.edu.cn
% Date: 9/15/2020
if size(pred_map, 1) ~= size(gt_map, 1) || size(pred_map, 2) ~= size(gt_map, 2)
    error('Saliency map and gt Image have different sizes!\n');
end

if ~islogical(gt_map)
    error('GT should be of type: logical');
end

pred_map(pred_map>=0.5) = 1; pred_map(pred_map<0.5) = 0; pred_map = logical(pred_map);

%compute true_positives
true_pos = (pred_map.*gt_map);
TP_count = sum(true_pos(:)); 

%compute true_negatives
true_neg = ((~pred_map).*(~gt_map));
TN_count = sum(true_neg(:)); 
 
%compute false_positives
FP_count = sum(pred_map(:)) - TP_count;
        
%compute false negatives
FN_count = sum(gt_map(:)) - TP_count;   

N_p = sum(gt_map(:));
N_n = sum(~gt_map(:));
N = N_p + N_n;

recall_p = TP_count / N_p;
recall_n = TN_count / N_n;
ber = 100 * (1 - 0.5 * (recall_p + recall_n));
