% MLP Final Version
% Programmed by Angus Hamilton

% Importing all scripts and functions from path
addpath('/home/ajh/Documents/Data Science/Biologically-Inspired Computation/bbob.v15.03/matlab');

function inputs = inputfinder(DIMN)
  % This function produces 15 inputs of a specified dimension
  input = 10 * rand(DIMN,15) - 5;
  inputs = input;
  end
 

function desiredval = dfinder(benchinput, benchfuncID)
  % This function finds the correct output for a benchmark function given
  % a vector of 2 inputs
  desiredval = benchmarks(benchinput, benchfuncID);
  end

function error = errfinder(desval, actval)
  % This function compares the desired value with the actual value to get
  % the error of the neural network
  error = desval - actval;
  end

function [chromosome, w0, w1, w2, w3] = chromeassembler(DIMN)
  % This function creates a chromosome containing 4 random weight matrixes that have as
  % many neurons as there are dimensions
  w0 = 10 * rand(DIMN,3) - 5;
  w1 = 10 * rand(DIMN,3) - 5;
  w2 = 10 * rand(DIMN,3) - 5;
  w3 = 10 * rand(DIMN,3) - 5;
  chromosome = [w0(:) ; w1(:) ; w2(:) ; w3(:)];
  end

function [w0,w1,w2,w3] = chromesplit(DIMN, chrome)
  % This is a helper function to split a chromosome into its component weight
  % matrices
  w00 = chrome(1:(DIMN*3));
  w01 = chrome((DIMN*3+1):(DIMN*3*2));
  w02 = chrome((DIMN*3*2+1):(DIMN*3*3));
  w03 = chrome((DIMN*3*3+1):(DIMN*3*4));
  w0 = [w00(1:DIMN),w00(DIMN+1:DIMN*2),w00(DIMN*2+1:DIMN*3)];
  w1 = [w01(1:DIMN),w01(DIMN+1:DIMN*2),w01(DIMN*2+1:DIMN*3)];
  w2 = [w02(1:DIMN),w02(DIMN+1:DIMN*2),w02(DIMN*2+1:DIMN*3)];
  w3 = [w03(1:DIMN),w03(DIMN+1:DIMN*2),w03(DIMN*2+1:DIMN*3)];
  end

function [actf] = activationfinder()
  % This creates a random activation function
  actf = 10 * rand(1) - 5;
  end 

function [plusarr, minusarr, actfp, actfm] = hillfinder(w0,w1,w2,w3)
  % This is a hillclimbing function that alters a random point in each
  % weight matrix, as well as the activation function
  actf = activationfinder();
  s = size(w0);
  r1 = randi(s(1));
  r2 = randi(s(2));
  % Weight 0
  addw01 = w0;
  minw01 = w0;
  addw02 = w0;
  minw02 = w0;
  addw03 = w0;
  minw03 = w0;
  % Weight 1
  addw11 = w1;
  minw11 = w1;
  addw12 = w1;
  minw12 = w1;
  addw13 = w1;
  minw13 = w1;
  % Weight 2
  addw21 = w2;
  minw21 = w2;
  addw22 = w2;
  minw22 = w2;
  addw23 = w2;
  minw23 = w2;
  % Weight 3
  addw31 = w3;
  minw31 = w3;
  addw32 = w3;
  minw32 = w3;
  addw33 = w3;
  minw33 = w3;
  % Weight 0 add and minus
  addw01(r1,r2) = w0(r1,r2) .+ 0.1;
  minw01(r1,r2) = w0(r1,r2) .- 0.1;
  addw02(r1,r2) = w0(r1,r2) .+ 0.5;
  minw02(r1,r2) = w0(r1,r2) .- 0.5;
  addw03(r1,r2) = w0(r1,r2) .+ 1.0;
  minw03(r1,r2) = w0(r1,r2) .- 1.0;
  % Weight 1 add and minus
  addw11(r1,r2) = w1(r1,r2) .+ 0.1;
  minw11(r1,r2) = w1(r1,r2) .- 0.1;
  addw12(r1,r2) = w1(r1,r2) .+ 0.5;
  minw12(r1,r2) = w1(r1,r2) .- 0.5;
  addw13(r1,r2) = w1(r1,r2) .+ 1.0;
  minw13(r1,r2) = w1(r1,r2) .- 1.0;
  % Weight 2 add and minus
  addw21(r1,r2) = w2(r1,r2) .+ 0.1;
  minw21(r1,r2) = w2(r1,r2) .- 0.1;
  addw22(r1,r2) = w2(r1,r2) .+ 0.5;
  minw22(r1,r2) = w2(r1,r2) .- 0.5;
  addw23(r1,r2) = w2(r1,r2) .+ 1.0;
  minw23(r1,r2) = w2(r1,r2) .- 1.0;
  % Weight 3 add and minus
  addw31(r1,r2) = w3(r1,r2) .+ 0.1;
  minw31(r1,r2) = w3(r1,r2) .- 0.1;
  addw32(r1,r2) = w3(r1,r2) .+ 0.5;
  minw32(r1,r2) = w3(r1,r2) .- 0.5;
  addw33(r1,r2) = w3(r1,r2) .+ 1.0;
  minw33(r1,r2) = w3(r1,r2) .- 1.0;
  pa0 = [addw01(:); addw11(:); addw21(:); addw31(:)];
  ma0 = [minw01(:); minw11(:); minw21(:); minw31(:)];
  pa1 = [addw02(:); addw12(:); addw22(:); addw32(:)];
  ma1 = [minw02(:); minw12(:); minw22(:); minw32(:)];
  pa2 = [addw03(:); addw13(:); addw23(:); addw33(:)];
  ma2 = [minw03(:); minw13(:); minw23(:); minw33(:)];
  plusarr = [pa0, pa1, pa2];
  minusarr = [ma0, ma1, ma2];
  actfp = [(actf+ 0.1),(actf + 0.5),(actf + 1.0)];
  actfm = [(actf- 0.1),(actf - 0.5),(actf - 1.0)];
  end


function [out, bestc,besta,totalerr] = steadytournament(in,DIMN,benchfuncID,trainingiter)
  % This is a steady state mutating tournament selection genetic algorithm to 
  % train the neural network
  ind = [];
  dval = dfinder(in, benchfuncID);
  created = 0;
  overc = [];
  overa = [];
  overerr = [];
  for iter = 1:trainingiter
    overr = [];
    contestc = [];
    contesta = [];
    contesterr = [];
    olderr = 100000000000000000;
    if created != 1
      for i2 = 1:50
        c = chromeassembler(DIMN);
        a = activationfinder();
        [fact,err] = mlprun(in,DIMN,c,a,benchfuncID);
        overc = [overc, c];
        overa = [overa, a];
        overerr = [overerr, err];
        created = 1;
      endfor
    endif
    [sortederr,ind] = sort(overerr);
    sumerr = sum(sortederr);
    for i3 = 1:5
      r = randi(50);
      overr = [overr, r];
    endfor
    for i4 = 1:2
      contestc = [contestc, overc(:,overr(i4))];
      contesta = [contesta, overa(overr(i4))];
      contesterr = [contesterr, overerr(overr(i4))];
    endfor
    [serr,indexes] = sort(contesterr);
    winner = [contesta(indexes(1)); contestc(:,indexes(1))];
    si = size(winner);
    mutationpoint = randi(si(1));
    mp = winner;
    mm = winner;
    mp(mutationpoint) = mp(mutationpoint) + 0.5;
    mm(mutationpoint) = mm(mutationpoint) - 0.5;
    [pfact,perr] = mlprun(in, DIMN, mp(2:end), mp(1),benchfuncID);
    [mfact,merr] = mlprun(in, DIMN, mm(2:end), mm(1),benchfuncID);
    if perr < max(overerr) && perr < merr
      overc(:,ind(end)) = mp(2:end);
      overa(ind(end)) = mp(1);
      overerr(ind(end)) = perr;
    elseif merr < max(overerr) && merr < perr
      overc(:,ind(end)) = mm(2:end);
      overa(ind(end)) = mm(1);
      overerr(ind(end)) = merr;
    endif
    [finalsorerr,ind] = sort(overerr);
    sumerr = sum(finalsorerr);
    besterr = finalsorerr(1);
    if sumerr == olderr
      overc = overc(:,ind(1));
      overa = overa(ind(1));
      overerr = overerr(1);
      for i2 = 1:49
        c = chromeassembler(DIMN);
        a = activationfinder();
        [fact,err] = mlprun(in,DIMN,c,a,benchfuncID);
        overc = [overc, c];
        overa = [overa, a];
        overerr = [overerr, err];
        created = 1;
      endfor
    endif
    if besterr < 15
      [bestout,besterr] = mlprun(in,DIMN,overc(:,ind(1)),overa(ind(1)),benchfuncID);
      bestc = overc(:,ind(1));
      besta = overa(ind(1));
    endif
    olderr = besterr;
  endfor
  [sortederr, ind] = sort(overr);
  [out, totalerr] = mlprun(in, DIMN, overc(:,ind(1)),overa(ind(1)),benchfuncID);
  bestc = overc(:,ind(1));
  besta = overa(ind(1));
  end

function [forwardprop,totalerr, inderr] = mlprun(in,DIMN,chrome,actf,benchfuncID)
  % This is a function that calculates the output, total error, and individual
  % error given multiple inputs, in a 4 layer neural network
  s = size(in);
  forwardprop = [];
  inderr = [];
  dvals = dfinder(in,benchfuncID);
  for sel = 1:s(2)
    inp = in(:,sel);
    act0 = [];
    act1 = [];
    act2 = [];
    BIAS = 1;
    overalloutput = 0;
    [w0,w1,w2,w3] = chromesplit(DIMN , chrome);
    in0 = (w0 .* inp) + BIAS;
    for i = 1:DIMN
      act0 =  [act0 ; actf*(sum(in0(i,:)))];
    endfor
    in1 = w1 .* act0;
    for i = 1:DIMN
      act1 =  [act1 ; actf*(sum(in1(i,:)))];
    endfor
    in2 = w2 .* act1;
    for i3 = 1:DIMN
      act2 =  [act2 ; actf*(sum(in2(i3,:)))];
    endfor
    in3 = w3 .* act2;
    for i4 = 1:DIMN
      overalloutput = overalloutput + (actf*(sum(in3(i4,:))));
    endfor
    forwardprop = [forwardprop, overalloutput];
    err = abs(errfinder(dvals(sel),overalloutput));
    inderr = [inderr, err];
  endfor
  totalerr = sum(inderr);
  end 


function [out,d, chrome,act, sumerr] = hilltrain(DIMN, in,benchfuncID)
  % This is an augmented hillclimbing algorithm that has 6 potential points
  % to move to
  chrome = chromeassembler(DIMN);
  a = activationfinder();
  act = a;
  out = 0;
  sumerr = Inf;
  d = dfinder(in,benchfuncID);
  for i = 1:1000
    [w0,w1,w2,w3] = chromesplit(DIMN, chrome);
    [plusarr,minusarr,actfp, actfm] = hillfinder(w0,w1,w2,w3);
    [forprop1,er1] = mlprun(in, DIMN, plusarr(:,1),actfp(1), DIMN,benchfuncID);
    [forprop2,er2] = mlprun(in, DIMN, minusarr(:,1),actfm(1), DIMN,benchfuncID);
    [forprop3,er3] = mlprun(in, DIMN, plusarr(:,2),actfp(2), DIMN,benchfuncID);
    [forprop4,er4] = mlprun(in, DIMN,minusarr(:,2),actfm(2), DIMN,benchfuncID);
    [forprop5,er5] = mlprun(in, DIMN,plusarr(:,3),actfp(3), DIMN,benchfuncID);
    [forprop6,er6] = mlprun(in, DIMN,minusarr(:,3),actfm(3), DIMN,benchfuncID);
    err1 = sum(abs(er1));
    err2 = sum(abs(er2));
    err3 = sum(abs(er3));
    err4 = sum(abs(er4));
    err5 = sum(abs(er5));
    err6 = sum(abs(er6));
    if (err1 == min([err1,err2,err3,err4,err5,err6]) && err1 < sumerr)
      act = actfp(1);
      chrome = plusarr(:,1);
      out = forprop1;
      sumerr = err1;
    elseif (err2 == min([err1,err2,err3,err4,err5,err6]) && err2 < sumerr)
      act = actfm(1);
      chrome = minusarr(:,1);
      out = forprop2;
      sumerr = err2;
    elseif (err3 == min([err1,err2,err3,err4,err5,err6]) && err3 < sumerr)
      act = actfp(2);
      chrome = plusarr(:,2);
      out = forprop3;
      sumerr = err3;
    elseif (err4 == min([err1,err2,err3,err4,err5,err6]) && err4 < sumerr)
      act = actfm(2);
      chrome = minusarr(:,2);
      out = forprop4;
      sumerr = err4;
    elseif (err5 == min([err1,err2,err3,err4,err5,err6]) && err5 < sumerr)
      act = actfp(3);
      chrome = plusarr(:,3);
      out = forprop5;
      sumerr = err5;
    elseif (err6 == min([err1,err2,err3,err4,err5,err6]) && err6 < sumerr)
      act =  actfm(3);
      chrome = minusarr(:,3);
      out = forprop6;
      sumerr = err6;
    else
      %Do nothing; Try again; Move back to start and lose 500 dollars
    endif
  endfor 
  end 


function [out,in,bestc,besta,totalerr] = main(in,DIMN, benchfuncID,trainingiter)
  % The function to end all functions; Trains an MLP on a DIMN benchmark function
  % for a certain number of generations using a steady tournament algorithm
  [out, bestc,besta,totalerr] = steadytournament(in,DIMN,benchfuncID,trainingiter);
  end
  