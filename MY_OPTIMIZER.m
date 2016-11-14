function [xbest,fprop,comp] = MY_OPTIMIZER(FUN, DIM, ftarget, maxfunevals)
% MY_OPTIMIZER(FUN, DIM, ftarget, maxfunevals)
% Samples points and performs a steadystate tournament selection algorithm with
% crossover at a random point.  In addition, this also trains an MLP in each 
% instance and runs the best input of each cycle on it, providing an approximate
% fvalue.  Note that this algorithm takes quite a long time to implement.
  
  maxfunevals = min(1e8 * DIM, maxfunevals); 
  popsize = min(maxfunevals, 200);
  for iter = 1:ceil(maxfunevals/popsize)
    overr = [];
    contest = [];
    if iter == 1
      xpop = 10 * rand(DIM, popsize) - 5;     % new solutions
      [out,d, bestc,besta, sumerr] = hilltrain(DIM, xpop,FUN);
      s = size(xpop(:,1));
      crosspoint = randi(s(1));
    endif
    for i = 1:5
      r = randi(popsize);
      overr = [overr, r];
    endfor
    for i2 = 1:5
      contest = [contest, xpop(:,overr(i2))];
    endfor
    [fvalues, idx] = sort(feval(FUN, xpop));
    fworst = fvalues(end);
    [contfvalues, contidx] = sort(feval(FUN, contest)); % evaluate
    t1 = contest(:,contidx(1));
    t2 = contest(:,contidx(2));
    new = [t1(1:crosspoint,1); t2(crosspoint+1:end,1)];
    newfvalue = feval(FUN,new);
    if fworst > newfvalue        % keep best
      xpop(:,idx(end)) = new;
    end
    [fvalues, idx] = sort(feval(FUN, xpop));
    xbest = xpop(:,idx(1));
    fprop = mlprun(xbest,DIM,bestc,besta,FUN); 
    fbest = fvalues(1);
    comp = abs(fbest) - abs(fprop);
    if feval(FUN, 'fbest') < ftarget         % COCO-task achieved
      break;                                 % (works also for noisy functions)
    end
  end 

  
