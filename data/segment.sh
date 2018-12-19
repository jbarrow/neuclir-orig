# $1 - Language 1 (swa | tgl | som)
# $2 - Input directory
# $3 - Output Directory

echo "Segmenting $1-language files located in $2\n"

morph=/storage/proj/ramy/morphology/latest/
analyzer=$morph/scripts-morph-v4.1.jar

mkdir $2
cd $morph

for directory in "ANALYSIS1" "ANALYSIS2" "DEV" "EVAL1" "EVAL2" "EVAL3"
do
  for doctype in "audio" "text"
  do
    echo " * Running on documents in $2/$directory/$doctype"
    echo " * Outputting results to $3/$directory/$doctype"
    mkdir -p $3/$directory/$doctype
    java -jar $analyzer $1 $2/$directory/$doctype $3/$directory/$doctype
  done
done
