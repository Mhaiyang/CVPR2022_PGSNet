clc;clear;

algorithms = {
    'PGSNet';
    };

datasets = {
              'test';
    };

tic
for i = 1:numel(algorithms)
    alg = algorithms{i};
    fprintf('%s\n', alg);
    txt_path = ['./mat/' alg '/'];
    if ~exist(txt_path, 'dir'), mkdir(txt_path); end
    fileID = fopen([txt_path 'results.txt'],'w');
    
    for j = 1:numel(datasets)
        dataset      = datasets{j};

        maskpath     = ['../data/RGBP-Glass/test/mask/'];
        predpath     = ['../results/' alg '/'];

        if ~exist(predpath, 'dir'), continue; end

        names = dir([maskpath '*.png']);
        names = {names.name}';
        iou          = 0; wf    = 0; mae     = 0; ber     = 0; recall_p     = 0; recall_n     = 0;
        score1       = 0; score2 = 0; score3 = 0; score4 = 0; score5 = 0; score6 = 0;

        results      = cell(numel(names), 6);
        file_num     = false(numel(names), 1);
        
        for k = 1:numel(names)
            name          = names{k,1};
            results{k, 1} = name;
            file_num(k)   = true;
            
            fgpath        = [predpath name];
            fg            = imread(fgpath);

            gtpath = [maskpath name];
            gt = imread(gtpath);

            if length(size(fg)) == 3, fg = fg(:,:,1); end
            if length(size(gt)) == 3, gt = gt(:,:,1); end
            
            fg = imresize(fg, size(gt)); 
            fg = mat2gray(fg); 
            gt = mat2gray(gt);
            
            gt(gt>=0.5) = 1; gt(gt<0.5) = 0; gt = logical(gt);
            
            score1                   = IoU(fg, gt);
            score2                   = wFmeasure(fg, gt); 
            score3                   = MAE(fg, gt);
            [score4, score5, score6] = BER(fg, gt);
            iou                      = iou + score1;
            wf                       = wf  + score2;
            mae                      = mae + score3;
            ber                      = ber + score4;
            recall_p                 = recall_p + score5;
            recall_n                 = recall_n + score6;
            results{k, 2}            = score1; 
            results{k, 3}            = score2; 
            results{k, 4}            = score3; 
            results{k, 5}            = score4;
            results{k, 6}            = score5;
            results{k, 7}            = score6;

        end

        file_num = double(file_num);
        iou      = iou / sum(file_num);
        wf       = wf  / sum(file_num);
        mae      = mae / sum(file_num);
        ber      = ber / sum(file_num);
        recall_p = recall_p / sum(file_num);
        recall_n = recall_n / sum(file_num);
        
        fprintf(fileID, '%10s (%4d images): I:%6.2f, F:%6.3f, M:%6.3f, B:%6.2f, P:%6.3f, N:%6.3f\n', dataset, sum(file_num), iou, wf, mae, ber, recall_p, recall_n);
        fprintf('%10s (%4d images): I:%6.2f, F:%6.3f, M:%6.3f, B:%6.2f, P:%6.3f, N:%6.3f\n', dataset, sum(file_num), iou, wf, mae, ber, recall_p, recall_n);
        save_path = ['./mat' filesep alg filesep];
        if ~exist(save_path, 'dir'), mkdir(save_path); end
        save([save_path 'results.mat'], 'results');
        

    end
end
toc
