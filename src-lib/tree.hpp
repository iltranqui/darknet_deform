#pragma once

#include "darknet_internal.hpp"

namespace Darknet
{
	struct Tree
	{
		int *leaf;
		int n;
		int *parent;
		int *child;
		int *group;
		VStr names;  ///< Class names (converted from char** to VStr)

		int groups;
		int *group_size;
		int *group_offset;
	};

	Tree *read_tree(const char * filename);
	Tree *read_tree(const std::string & filename);

	int hierarchy_top_prediction(float *predictions, Tree *hier, float thresh, int stride);
	void hierarchy_predictions(float *predictions, int n, Tree *hier, int only_leaves);
	float get_hierarchy_probability(float *x, Tree *hier, int c);
}
