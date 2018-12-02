% Generate a sample of size 10,000 from the uniform distribution of support 100,000
n=100000; k=10000;
samp = randi(n,k,1);


% Compute corresponding 'fingerprint'
f = makeFinger(samp);


% Estimate histogram of distribution from which sample was drawn
[h,x]=unseen(f);


%output entropy of the true distribution, Unif[n]
trueEntropy = log(n)

%output entropy of the empirical distribution of the sample
empiricalEntropy = -f'*(((1:max(size(f)))/k).*log(((1:max(size(f)))/k)))'

%output entropy of the recovered histogram, [h,x]
estimatedEntropy = -h*(x.*log(x))'

%output entropy using entropy_estC.m (should be almost the same as above):
estimatedEntropy2 = entropy_estC(f)


