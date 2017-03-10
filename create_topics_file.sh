#!/usr/bin/env bash
cat ../../tdt2_em_v4_0/doc/topics/tdt2_topic_rel.complete_annot.v3.3 | grep "ONTOPIC" | cut -d" " -f 2| cut -d"=" -f 2 | cut -c 3- > topics_list.tmp
cat ../../tdt2_em_v4_0/doc/topics/tdt2_topic_rel.complete_annot.v3.3 | grep "ONTOPIC" | cut -d" " -f 4| cut -d"=" -f 2 > story_list.tmp
cat ../../tdt2_em_v4_0/doc/topics/tdt2_topic_rel.complete_annot.v3.3 | grep "ONTOPIC" | cut -d" " -f 5| cut -d"=" -f 2 > file_list.tmp
cat ../../tdt2_em_v4_0/doc/topics/tdt2_topic_rel.complete_annot.v3.3 | grep "ONTOPIC" | cut -d" " -f 3| cut -d"=" -f 2 > relevance_list.tmp

paste story_list.tmp file_list.tmp topics_list.tmp relevance_list.tmp | column -s $'\t' -t > story_topics.tbl
rm -rf relevance_list.tmp file_list.tmp story_list.tmp topics_list.tmp
