sptreat_loop <- function(dat_Y,dat_D,dat_S,w_fun="bin",dd,hh,maxdeffect,show.progress=FALSE) {
  if (length(hh) == 1) {
    hh <- rep(hh,times=length(dd))
  }
  if (length(dd) != length(hh)) { stop("hh must be either length 1 or the same length as dd") }
  
  if (w_fun %in% c("uniform","bin","uni")) {
    w_fun <- function(d,d_estimand,h,i=NULL,s=NULL) {
      return(as.numeric((d>d_estimand-h/2) & (d<=d_estimand+h/2)))
    }
  } else if (w_fun %in% c("tri","triangular")) {
    w_fun <- function(d,d_estimand,h,i=NULL,s=NULL) {
      return( (1 - abs(d_estimand - d)/h) * (abs(d_estimand-d)<=h) )
    }
  }
  
  dat_out <- data.frame()
  for (b in seq(dd)) {
    if (show.progress==TRUE) { cat(paste("next:",b,"/",length(dd),Sys.time(),"\n")) }
    dat_w <- dat_D
    dat_w$w <- w_fun(dat_D$d,dd[b],hh[b],dat_D$i,dat_D$s)
    dat_w <- dat_w[dat_w$w!=0,]
    dat_out <- rbind(dat_out,
                     cbind(dist_bin=b-1,
                           d = dd[b],
                           as.data.frame(estimate_sptreat(dat_D=dat_D,
                                                          dat_S = dat_S,
                                                          dat_Y = dat_Y,
                                                          dat_w = dat_w,
                                                          maxdeffect = maxdeffect))))
                             
  }
  return(dat_out)
}


estimate_sptreat <- function(dat_D,dat_S,dat_Y,dat_w,maxdeffect) {
  # TODO: input checks
  
  # 
  dt.w <- as.data.table(dat_w)
  dt.Y <- as.data.table(dat_Y)
  dt.S <- as.data.table(dat_S)
  S <- nrow(dt.S)
  
  dt.res <- (
    dt.w
    [dt.Y, on=.(i), nomatch=NULL]
    [dt.S, on=.(s), nomatch=NULL]
    [,.(mut = sum(w*W*Y)/sum(w*W), muc=sum(w*(1-W)*(p/(1-p))*Y)/sum(w*(1-W)*(p/(1-p))),
        nbart = sum(w*W)/S, nbarc = sum(w*(1-W)*p/(1-p))/S, nbar = sum(w*p)/S)]
  )
  mut <- ifelse(is.finite(dt.res$mut),dt.res$mut,0)
  muc <- ifelse(is.finite(dt.res$muc),dt.res$muc,0)
  
  nbar <- dt.res$nbar
  nbart <- ifelse(dt.res$nbart>0, dt.res$nbart, nbar)
  nbarc <- ifelse(dt.res$nbarc>0, dt.res$nbarc, nbar)
  
  
  V <- estimate_variance(dat_D=dat_D,dat_S=dat_S,dat_Y=dat_Y,dat_w=dat_w,
                         muhat = c(muc,mut), nbar = c(nbar,nbar),#c(nbarc,nbart),
                         maxdeffect=maxdeffect)
  
  return(list(tau_hat=mut-muc,se=sqrt((V$Vt+V$Vc+V$Vx)/S),
           muc=muc,mut=mut,nbart=dt.res$nbart,nbarc=dt.res$nbarc,nbar=nbar,V=V$V/S,Vt=V$Vt/S,Vc=V$Vc/S,Vx=V$Vx/S))
}



sourceCpp("sptreat_var.cpp")

# estimator of variance
estimate_variance <- function(dat_D,dat_S,dat_Y,dat_w,muhat,nbar,maxdeffect) {
  
  S_orig <- nrow(dat_S)
  
  dat_w <- dat_w[dat_w$w!=0,]
  I_val <- sort(unique(dat_w$i[dat_w$i %in% dat_Y$i]))
  dat_w <- dat_w[dat_w$i %in% I_val,]
  dat_D <- dat_D[(dat_D$i %in% I_val) & (dat_D$d <= maxdeffect),]
  dat_Y <- dat_Y[dat_Y$i %in% I_val,]
  
  # either non-zero weight for this s or it affects someone with non-zero weight
  S_val <- unique(c(dat_w$s[dat_w$w!=0],
                    dat_D$s[dat_D$d<=maxdeffect]))
  S <- length(S_val)  #length(unique(dat_w$s))
  dat_S_crosswalk <- data.frame(s=sort(S_val), s_new=0:(S-1))
  
  dat_D <- merge(dat_D,dat_S_crosswalk, by="s")
  dat_D$s <- dat_D$s_new
  dat_S <- merge(dat_S,dat_S_crosswalk, by="s")
  dat_S$s <- dat_S$s_new
  dat_w <- merge(dat_w,dat_S_crosswalk, by="s")
  dat_w$s <- dat_w$s_new
  
  
  dat_I_crosswalk <- data.frame(i=I_val, i_new=0:(length(I_val)-1))
  dat_D <- merge(dat_D,dat_I_crosswalk, by="i")
  dat_D$i <- dat_D$i_new
  dat_Y <- merge(dat_Y,dat_I_crosswalk, by="i")
  dat_Y$i <- dat_Y$i_new
  dat_w <- merge(dat_w,dat_I_crosswalk, by="i")
  dat_w$i <- dat_w$i_new
  
  
  p <- dat_S$p
  # setup
  dt.D <- as.data.table(dat_D[,c("i","s","d")])
  setkey(dt.D,i,s)
  # dt.D[d<=maxdeffect]
  dt.S <- as.data.table(dat_S[,c("s","p","W")])[,.(s,p,W=as.numeric(W))]
  setkey(dt.S,s)
  dt.Y <- as.data.table(dat_Y[,c("i","Y")])
  setkey(dt.Y,i)
  dt.w <- as.data.table(dat_w[dat_w$w!=0,c("i","s","d","w")])
  setkey(dt.w,i,s)
  
  # filter to relevant locations, merge probs., aggregate probs., add i with no relevant locations
  dt.pm <- dt.D[d<=maxdeffect][dt.S, on=.(s)][,.(pm=prod(p^W*(1-p)^(1-W))),by=.(i)][dt.Y[,.(i)], on=.(i)]
  setnafill(dt.pm,fill=1)  # if no relevant location, prob of this exposure = 1
  setkey(dt.pm,i)

  # p <- dat_S$p
  # setup
  dt.D <- as.data.table(dat_D[,c("i","s","d")])
  setkey(dt.D,i,s)
  # dt.D[d<=maxdeffect]
  dt.S <- as.data.table(dat_S[,c("s","p","W")])[,.(s,p,W=as.numeric(W))]
  setkey(dt.S,s)
  dt.Y <- as.data.table(dat_Y[,c("i","Y")])
  setkey(dt.Y,i)
  dt.w <- as.data.table(dat_w[dat_w$w!=0,c("i","s","d","w")])
  setkey(dt.w,i,s)
  
  dt.out <- dt.w[dt.S,nomatch=NULL,on=.(s)][dt.Y, nomatch=NULL, on=.(i)]
  
  allExposures <- function(I_val,dat_D,dat_S) {
    M <- list()
    for (i in I_val) {
      # which locations can affect i
      Si <- sort(dat_D$s[(dat_D$i == i) & (dat_D$d <= maxdeffect)])
      # which individuals are affected by one of those locations
      Iconflict <- sort(unique(dat_D$i[(dat_D$s %in% Si) & (dat_D$d <= maxdeffect)]))
      
      if (i==54) {
        a <- 5;
      }
      
      if (length(Si)>0) {
        # what are the possible treatment assignments
        m <- as.matrix(expand.grid(rep(list(c(0,1)),length(Si))))
        # what are the probabilities of those assignments
        pm <- exp(m %*% log(dat_S$p[Si+1]) + (1-m)%*%log(1-dat_S$p[Si+1]))
      } else {
        m <- matrix(NA,nrow=1,ncol=0)
        pm <- 1
      }
      m_realized <- which(apply(m,1,function(a) all(a==dat_S$W[Si+1], na.rm=TRUE))) - 1 # -1 because c++ index starts at 0
      
      # pmsa <- array(dim=c(nrow(m),S,2))
      pmsa0 <- matrix(nrow=nrow(m),ncol=S)
      pmsa1 <- matrix(nrow=nrow(m),ncol=S)
      for (s in 1:S) {
        s_idx <- match(s-1,Si)  # -1 because index starts at 0 for c++
        if (!is.na(s_idx)) {
          # pmsa[,s,1+FALSE] <- pm * (1-m[,s_idx])
          # pmsa[,s,1+TRUE] <- pm * m[,s_idx]
          pmsa0[,s] <- pm * (1-m[,s_idx])
          pmsa1[,s] <- pm * m[,s_idx]
        } else {
          # pmsa[,s,1+FALSE] <- pm*(1-dat_S$p[s])
          # pmsa[,s,1+TRUE] <- pm*dat_S$p[s]
          pmsa0[,s] <- pm*(1-dat_S$p[s])
          pmsa1[,s] <- pm*dat_S$p[s]
        }
      }
      
      M[[paste0("i",i)]]$i <- i
      M[[paste0("i",i)]]$S <- Si
      M[[paste0("i",i)]]$Iconflict <- Iconflict
      M[[paste0("i",i)]]$m <- m
      M[[paste0("i",i)]]$m_realized <- m_realized
      M[[paste0("i",i)]]$p <- pm
      M[[paste0("i",i)]]$w <- dat_w[dat_w$i==i,c("s","w")]
      # M[[paste0("i",i)]]$pmsa <- pmsa
      M[[paste0("i",i)]]$pmsa0 <- pmsa0
      M[[paste0("i",i)]]$pmsa1 <- pmsa1
    }
    return(M)
  }
  M <- allExposures(I_val=sort(unique(dat_Y$i)),dat_D=dat_D,dat_S=dat_S)
  
  Va <- calc_Va(dt_out=as.matrix(dt.out), p=dat_S$p, M=M, S=S, muhat=muhat, nbar=nbar)

  
  
  # second term needs conflicting exposures
  dt.overlap <- (
    dt.D[d<=maxdeffect][,.(i1=i,s=s)]  # (i,s) with possible effect
    [dt.D[d<=maxdeffect][,.(i2=i,s=s)], on=.(s), allow.cartesian=TRUE]  # find all that overlap
    [dt.S[,.(s,p,W)], on=.(s), nomatch=NULL]  # merge in state and probability of the location
    [,.(overlap_p=prod(p^W*(1-p)^(1-W))),by=.(i1,i2)]  # remove duplicates if multiple overlap
  )
  setkey(dt.overlap,i1,i2)
  
  # Vx and Vo terms
  # (i,s) pairs that either determine exposure or have non-zero weight
  dt.is <- unique(rbind(dt.D[d<=maxdeffect][,.(i1=i,s=s)], dt.w[,.(i1=i,s=s)]))
  setkey(dt.is,s)
  # overlap between individuals
  dt.pmm <- (
    dt.overlap
    [unique(dt.is[dt.is,.(i1,i2=i.i1), allow.cartesian=TRUE])]  # all relevant (i1,s,i2)
    [dt.pm[,.(i1=i,pm1=pm)], on=.(i1)]  # marginal probabilities
    [dt.pm[,.(i2=i,pm2=pm)], on=.(i2)]
    # calculate joint probability
    [,c(.SD,.(pmm=pm1*pm2/fifelse(!is.na(overlap_p),overlap_p,1)))]  # remove overlap probability
    [,.(i1,i2,pm1,pm2,pmm)]
  )
  setkey(dt.pmm,i1,i2)
  
  dt.x <- (
    dt.pmm
    # outcomes
    [dt.Y[,.(i1=i,Y1=Y)], on=.(i1), nomatch=NULL]
    [dt.Y[,.(i2=i,Y2=Y)], on=.(i2), nomatch=NULL]
    # (s1,s2) pairs for these (i1,i2)
    [dt.w[,.(i1=i,s1=s,w1=w)], on=.(i1), allow.cartesian=TRUE, nomatch=NULL]
    [dt.S[,.(s1=s,p1=p,W1=W)], on=.(s1), nomatch=NULL]
    [dt.w[,.(i2=i,s2=s,w2=w)], on=.(i2), allow.cartesian=TRUE, nomatch=NULL]
    [dt.S[,.(s2=s,p2=p,W2=W)], on=.(s2), nomatch=NULL]
  )
  setkey(dt.x,i1,i2,s1,s2)
  dt.x[dt.D[d<=maxdeffect][,.(i,s,isnear=1)],on=.(i1=i,s1=s),i1s1 := i.isnear]
  dt.x[dt.D[d<=maxdeffect][,.(i,s,isnear=1)],on=.(i1=i,s2=s),i1s2 := i.isnear]
  dt.x[dt.D[d<=maxdeffect][,.(i,s,isnear=1)],on=.(i2=i,s1=s),i2s1 := i.isnear]
  dt.x[dt.D[d<=maxdeffect][,.(i,s,isnear=1)],on=.(i2=i,s2=s),i2s2 := i.isnear]
  setnafill(dt.x,fill=0)
  
  
  Vx <- 0
  for (WW1 in c(0,1)) {
    for (WW2 in c(0,1)) {
      Vx <- Vx + (
        dt.x
        # skip case where i1==i2, s1==s2, WW1==WW2
        [i1!=i2 | s1!=s2 | WW1!=WW2,]
        # if WW1, WW2 lead to marginal exposures that differ from the observed one
        #   then we need to set pp==0 below, easier to drop this case here
        [(W1 == WW1) | (i1s1==0),]
        [(W2 == WW2) | (i2s2==0),]
        # squared demeaned outcome
        [,c(.SD,.(Y1d=(Y1-muhat[1+WW1]), Y2d=(Y2-muhat[1+WW2])))]
        [,c(.SD,.(piimmssaa=fifelse((((W1 == WW1) | ((i1s1==0) & (i2s1==0)))
                                     & ((W2 == WW2) | ((i1s2==0) & (i2s2==0)))
                                     & ((WW1==WW2) | (s1!=s2))),
                                    pmm
                                    * (p1^WW1*(1-p1)^(1-WW1))^((1-i1s1)*(1-i2s1))
                                    * (p2^WW2*(1-p2)^(1-WW2))^((1-i1s2)*(1-i2s2)*(s1!=s2)),
                                    0)))]
        # the following line relies on WW1,WW2 not conflicting with marginal exposures
        #   alternatively would need to set the pm1 and pm2 products to 0
        [,c(.SD,.(pp=piimmssaa - (pm1  *(p1^WW1*(1-p1)^(1-WW1))^(1-i1s1)
                                  * pm2*(p2^WW2*(1-p2)^(1-WW2))^(1-i2s2))))]
        [,.(v = sum(pp * (-p1/(1-p1))^(1-WW1) * (-p2/(1-p2))^(1-WW2)
                    * w1/nbar[1+WW1] * w2/nbar[1+WW2] * Y1d * Y2d / pmm)/S_orig)]
      )$v
    }
  }
  
  return(list(V=(Va[1]+Va[2]+Vx)/S_orig,Vt=Va[2],Vc=Va[1], Vx=Vx))
}
