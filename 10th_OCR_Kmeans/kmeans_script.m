% K-means clustering sample in MATLAB

cluster_n = 5;
img_size = 512;

% Generate bright HSV palette as in Python (converted to RGB)
colors_hsv = zeros(1, cluster_n, 3, 'single');
colors_hsv(1,:,1) = linspace(0, 1 - 1/cluster_n, cluster_n);
colors_hsv(1,:,2) = 1;
colors_hsv(1,:,3) = 1;
colors_rgb = squeeze(hsv2rgb(colors_hsv));

fprintf('Press any key to update distributions, ESC to exit\n');

while true
    fprintf('Sampling distributions...\n');
    points = make_gaussians(cluster_n, img_size);
    
    % K-means clustering
    term_crit = [30, 0.1];
    [labels, ~, ~] = kmeans(points, cluster_n, 'MaxIter', term_crit(1), 'Replicates', 10, 'Display','off');

    % Prepare image for visualization
    img = zeros(img_size, img_size, 3, 'uint8');
    
    for i = 1:size(points,1)
        x = round(points(i,1));
        y = round(points(i,2));
        if x >= 1 && x <= img_size && y >=1 && y <= img_size
            color = uint8(colors_rgb(labels(i), :) * 255);
            for ch = 1:3
                img(y,x,ch) = color(ch);
            end
        end
    end
    
    imshow(img);
    title('kmeans');
    
    % Wait for key press
    k = waitforbuttonpress;
    ch = double(get(gcf,'CurrentCharacter'));
    if ch == 27 % ESC key ASCII
        break;
    end
end

fprintf('Done\n');

% --- Function that creates Gaussian mixtures ---
function points = make_gaussians(cluster_n, img_size)
    points_all = [];
    for i = 1:cluster_n
        mean_vec = (0.1 + 0.8 * rand(1,2)) * img_size;
        a = (rand(2,2) - 0.5) * img_size * 0.1;
        cov_mat = a' * a + img_size * 0.05 * eye(2);
        n = 100 + randi(900);
        pts = mvnrnd(mean_vec, cov_mat, n);
        points_all = [points_all; pts];
    end
    points = single(points_all);
end
