library(ggplot2)

## CONFIGURATION
ORDER <- c("RANDOM", "FOC", "MORLOT", "NSGA3")
LABELS <- c("RS", "FF", "MORLOT", "PF")
WD <- "my/absolute/path/to/out/directory" # replace here your path
setwd(WD)
##

# violations box plot
data <- read.csv("reqs_all.csv", header = TRUE, sep = ",")
data <- data[data$alg %in% ORDER, ]

first_row <- data[1, ]
df <- data.frame(first_row$alg, "R0", first_row$R0)
names(df) <- c("alg", "req", "val")

first_iter <- TRUE
for (i in 1:nrow(data)) {
    r <- data[i, ]
    if (!first_iter) {
        df[nrow(df) + 1, ] <- c(r$alg, "R0", r$R0)
    }
    df[nrow(df) + 1, ] <- c(r$alg, "R1", r$R1)
    df[nrow(df) + 1, ] <- c(r$alg, "R2", r$R2)
    df[nrow(df) + 1, ] <- c(r$alg, "R3", r$R3)
    df[nrow(df) + 1, ] <- c(r$alg, "R4", r$R4)
    df[nrow(df) + 1, ] <- c(r$alg, "R5", r$R5)
    first_iter <- FALSE
}

df$req = factor(df$req, levels=c('R0','R1','R2','R3', 'R4', 'R5'))

#df[df$req == "R0", ]
p <- ggplot(df, aes(x = alg, y = as.numeric(val))) +
    geom_boxplot(aes(fill = factor(alg, levels = ORDER))) + facet_wrap(~req, nrow = 1) +
    ylab("individual failures") +
    xlab(NULL) +
    labs(fill = "algorithm") +
    scale_x_discrete(limits = ORDER, labels = LABELS) +
    theme_bw() +
    theme(text = element_text(size = 14),
        axis.text.x = element_text(angle = 30, hjust = 1),
        aspect.ratio = 12 / 8,
        plot.margin=unit(c(-0.30,0,0,0), "null"),
        legend.position = "none")
    #ylim(c(0, 100))
#print(p)
ggsave(file = "imgs/violations.pdf", dpi = 300)
system(paste("pdfcrop --margins '0 0 0 0' imgs/violations.pdf", "imgs/b2_violations.pdf", sep = " "))


# score box plot
XLABELS <- c("x0", "x1", "x2", "x3", "x4")
data <- read.csv("score_all.csv", header = TRUE, sep = ",")
data <- data[data$alg %in% ORDER, ]

first_row <- data[1, ]
df <- data.frame(first_row$alg, "V0", first_row$V0)
names(df) <- c("alg", "variable", "score")

first_iter <- TRUE
for (i in 1:nrow(data)) {
    r <- data[i, ]
    if (!first_iter) {
        df[nrow(df) + 1, ] <- c(r$alg, "V0", r$V0)
    }
    df[nrow(df) + 1, ] <- c(r$alg, "V1", r$V1)
    df[nrow(df) + 1, ] <- c(r$alg, "V2", r$V2)
    df[nrow(df) + 1, ] <- c(r$alg, "V3", r$V3)
    df[nrow(df) + 1, ] <- c(r$alg, "V4", r$V4)
    first_iter <- FALSE
}

p <- ggplot(df, aes(x = variable, 
        y = as.numeric(score), 
        fill = factor(alg, levels = ORDER))) +
    geom_boxplot(width = 0.9) +
    #stat_summary(fun.y = mean, geom = "point", shape = 2, size = 3, color = "black",
    #         position = position_dodge2(width = 0.75, preserve = "single")) +
    ylab("score") +
    xlab("variable") +
    labs(fill = "algorithm") +
    scale_x_discrete(labels = XLABELS) +
    scale_fill_discrete(labels = LABELS) +
    scale_y_continuous(breaks = seq(-0.01, 0.02, 0.01)) +
    theme_bw() +
    theme(text = element_text(size = 32),
        axis.title.y = element_blank(),
        #axis.text.x = element_text(angle = 40, hjust = 1),
        aspect.ratio = 10 / 8,
        plot.margin = margin(0, 0, 0, 0),
        legend.title = element_blank(),
        legend.position = "none")
#print(p)
ggsave(file = "imgs/scores.pdf", dpi = 300)
system(paste("pdfcrop --margins '0 0 0 0' imgs/scores.pdf", "imgs/b2_scores.pdf", sep = " "))