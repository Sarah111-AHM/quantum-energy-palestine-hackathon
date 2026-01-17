# statistical_analysis.R
# Humanitarian Sites Analysis - Gaza

# ===== Setup =====
cat("================================\n")
cat("Humanitarian Sites Analysis - Gaza\n")
cat("================================\n")

# Install/load packages
pkgs <- c("dplyr", "ggplot2", "corrplot", "psych", "jsonlite")
new_pkgs <- pkgs[!(pkgs %in% installed.packages()[, "Package"])]
if(length(new_pkgs)) install.packages(new_pkgs)
lapply(pkgs, require, character.only = TRUE)

# ===== Load Data =====
load_data <- function(file = "../data/processed/processed_sites.csv") {
  cat("Loading data from:", file, "\n")
  
  if (!file.exists(file)) {
    cat("File not found. Using sample data.\n")
    set.seed(42); n <- 30
    data <- data.frame(
      id = 1:n,
      name = paste("Site", 1:n),
      region = sample(c("North Gaza","Gaza City","Middle","Khan Younis","Rafah"), n, TRUE),
      type = sample(c("Hospital","School","Camp","Aid Center"), n, TRUE),
      risk = round(runif(n, 0.2,0.9),3),
      access = round(runif(n, 0.3,0.9),3),
      priority = round(runif(n, 0.5,1),3),
      pop = sample(1000:20000, n, TRUE),
      cost = sample(50000:200000, n, TRUE)
    )
    data$base_score <- round(0.4*(1-data$risk) + 0.3*data$access + 0.3*data$priority,3)
    data$final_score <- round(data$base_score*0.8 + runif(n,0.6,0.9)*0.2,3)
    data$rank <- rank(-data$final_score)
    cat("Sample data generated for", n, "sites\n")
  } else {
    data <- read.csv(file, stringsAsFactors = FALSE)
    cat("Loaded", nrow(data), "records\n")
  }
  return(data)
}

# ===== Descriptive Stats =====
desc_analysis <- function(data) {
  cat("\nDescriptive Analysis\n-------------------\n")
  nums <- intersect(c("risk","access","priority","final_score","pop","cost"), names(data))
  if(length(nums)>0) print(describe(data[,nums])[,c("n","mean","sd","min","max","skew","kurtosis")])
  
  if("region" %in% names(data)) {
    cat("\nRegion distribution:\n")
    print(table(data$region))
    p <- ggplot(data,aes(x=region,fill=region)) + geom_bar() +
      labs(title="Sites by Region", x="Region", y="Count") +
      theme_minimal() + theme(axis.text.x=element_text(angle=45,hjust=1),legend.position="none") +
      scale_fill_brewer(palette="Set3")
    print(p)
  }
  
  if("type" %in% names(data)) {
    cat("\nSite type distribution:\n")
    print(table(data$type))
  }
}

# ===== Correlation =====
cor_analysis <- function(data) {
  cat("\nCorrelation Analysis\n------------------\n")
  nums <- intersect(c("risk","access","priority","final_score","pop","cost"), names(data))
  if(length(nums)>=2) {
    cor_mat <- cor(data[,nums],use="complete.obs")
    print(round(cor_mat,3))
    png("../results/analysis/corr_matrix.png",800,800)
    corrplot(cor_mat, method="color", type="upper", tl.col="black", tl.srt=45, addCoef.col="black", number.cex=0.8)
    dev.off()
    cat("Correlation matrix saved.\n")
    
    p <- ggplot(data,aes(x=risk,y=final_score,color=region)) +
      geom_point(size=3,alpha=0.7) + geom_smooth(method="lm",se=FALSE,color="red") +
      labs(title="Risk vs Final Score", x="Risk Score", y="Final Score") + theme_minimal() +
      scale_color_brewer(palette="Set1")
    print(p)
  }
}

# ===== PCA & Clustering =====
cluster_analysis <- function(data) {
  cat("\nPCA & Clustering\n----------------\n")
  nums <- intersect(c("risk","access","priority","final_score"), names(data))
  if(length(nums)>=2) {
    pca <- prcomp(data[,nums],scale=TRUE)
    cat("PCA summary:\n"); print(summary(pca))
    
    pca_df <- as.data.frame(pca$x); pca_df$region <- data$region
    p <- ggplot(pca_df,aes(x=PC1,y=PC2,color=region,shape=data$type)) +
      geom_point(size=4,alpha=0.7) +
      labs(title="PCA of Sites") + theme_minimal() + scale_color_brewer(palette="Set1")
    print(p)
    
    set.seed(42)
    km <- kmeans(scale(data[,nums]),3)
    data$cluster <- as.factor(km$cluster)
    cat("\nK-means clusters:\n"); print(table(data$cluster))
    
    p2 <- ggplot(data,aes(x=risk,y=final_score,color=cluster)) + geom_point(size=3) +
      labs(title="K-means Clustering", x="Risk", y="Final Score") + theme_minimal()
    print(p2)
  }
}

# ===== Geospatial Analysis =====
geo_analysis <- function(data) {
  cat("\nGeospatial Analysis\n-----------------\n")
  if(all(c("latitude","longitude") %in% names(data))) {
    p <- ggplot(data,aes(x=longitude,y=latitude)) +
      stat_density_2d(aes(fill=..level..),geom="polygon",alpha=0.5) +
      geom_point(aes(color=final_score,size=pop),alpha=0.7) +
      scale_color_gradient(low="blue",high="red") + theme_minimal() +
      labs(title="Site Density & Distribution",x="Longitude",y="Latitude")
    print(p)
  }
}

# ===== Cost Efficiency =====
cost_analysis <- function(data) {
  cat("\nCost Efficiency Analysis\n----------------------\n")
  if(all(c("cost","pop","final_score") %in% names(data))) {
    data$cost_per_person <- data$cost/data$pop
    data$efficiency <- data$final_score / data$cost_per_person * 1000
    top_eff <- data %>% arrange(desc(efficiency)) %>% head(10)
    cat("Top 10 efficient sites:\n"); print(top_eff[,c("name","region","final_score","cost","pop","cost_per_person","efficiency")])
    p <- ggplot(data,aes(x=cost_per_person,y=final_score,size=pop,color=region)) +
      geom_point(alpha=0.7) + geom_smooth(method="lm",se=FALSE,color="black",linetype="dashed") +
      labs(title="Cost vs Final Score",x="Cost per Person ($)",y="Final Score") + theme_minimal() +
      scale_color_brewer(palette="Set1")
    print(p)
  }
}

# ===== Save Reports =====
save_reports <- function(data) {
  dir.create("../results/analysis",showWarnings=FALSE,recursive=TRUE)
  write_json(list(n_sites=nrow(data),n_vars=ncol(data),date=Sys.time()), "../results/analysis/summary.json",pretty=TRUE)
  top_sites <- data %>% arrange(desc(final_score)) %>% head(20)
  write.csv(top_sites,"../results/analysis/top_sites.csv",row.names=FALSE)
  cat("Reports saved.\n")
}

# ===== Main =====
main <- function() {
  data <- load_data()
  desc_analysis(data)
  cor_analysis(data)
  cluster_analysis(data)
  geo_analysis(data)
  cost_analysis(data)
  save_reports(data)
  cat("\nAnalysis complete. Sites:", nrow(data), "\n")
}

# Run
main()
