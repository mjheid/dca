library(splatter)

simulate <- function(nGroups=2, nGenes=200, batchCells=2000, dropout=5)
{
    if (nGroups > 1) method <- 'groups'
    else             method <- 'single'

    group.prob <- rep(1, nGroups) / nGroups

    # new splatter requires dropout.type
    if ('dropout.type' %in% slotNames(newSplatParams())) {
        if (dropout)
            dropout.type <- 'experiment'
        else
            dropout.type <- 'none'
        
        sim <- splatSimulate(group.prob=group.prob, nGenes=nGenes, batchCells=batchCells,
                             dropout.type=dropout.type, method=method,
                             seed=42, dropout.shape=-1, dropout.mid=dropout)

    } else {
        sim <- splatSimulate(group.prob=group.prob, nGenes=nGenes, batchCells=batchCells,
                             dropout.present=!dropout, method=method,
                             seed=42, dropout.shape=-1, dropout.mid=dropout)        
    }

    counts     <- as.data.frame(t(counts(sim)))
    truecounts <- as.data.frame(t(assays(sim)$TrueCounts))

    dropout    <- as.matrix(assays(sim)$Dropout)
    mode(dropout) <- 'integer'
    dropout    <- as.data.frame(t(dropout))

    cellinfo   <- as.data.frame(colData(sim))
    geneinfo   <- as.data.frame(rowData(sim))

    list(counts=counts,
         cellinfo=cellinfo,
         geneinfo=geneinfo,
         truecounts=truecounts,
         dropout=dropout)
}

sim <- simulate(nGroups=6, dropout=3)

counts <- sim$counts
geneinfo <- sim$geneinfo
cellinfo <- sim$cellinfo
truecounts <- sim$truecounts
dropout <- sim$dropout

# write.csv(counts, '/home/kaies/csb/dca/data/fc/counts.csv')
# write.csv(geneinfo, '/home/kaies/csb/dca/data/fc/geneinfo.csv')
# write.csv(cellinfo, '/home/kaies/csb/dca/data/fc/cellinfo.csv')
# write.csv(truecounts, '/home/kaies/csb/dca/data/fc/truecounts.csv')
# write.csv(dropout, '/home/kaies/csb/dca/data/fc/dropout.csv')

write.csv(counts, '/home/kaies/csb/dca/data/sixgroupsimulation/counts.csv')
write.csv(geneinfo, '/home/kaies/csb/dca/data/sixgroupsimulation/geneinfo.csv')
write.csv(cellinfo, '/home/kaies/csb/dca/data/sixgroupsimulation/cellinfo.csv')
write.csv(truecounts, '/home/kaies/csb/dca/data/sixgroupsimulation/truecounts.csv')
write.csv(dropout, '/home/kaies/csb/dca/data/sixgroupsimulation/dropout.csv')